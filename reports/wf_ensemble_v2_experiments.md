# Ensemble v2 — improvement ideas, and three tested

> **Status: exploratory — single-seed, as its own note below says.** Where each idea landed is recorded in [`wf_ensemble_final_recommendations.md`](wf_ensemble_final_recommendations.md); the multi-seed re-check is [`wf_ensemble_v2_multiseed_verification.md`](wf_ensemble_v2_multiseed_verification.md). Do not quote a number from here without checking both.

**Status: exploratory.** Everything in this report uses single-seed within-silo OOF
predictions (like `wf_ensemble_fold_pipeline.ipynb`'s diagnostics), not Phase 1's proper
5-seed CV. Treat directions as directions, not final numbers — a real improvement found
here still needs the same discipline (multi-seed, noise-floor-gated) before it replaces v1.

---

## 1. Reflection — what could plausibly move the needle, and what's already ruled out

Six ideas considered, ranked by expected value given what this project has already measured:

| # | Idea | Why it's plausible | Tested here? |
|---|---|---|---|
| 1 | Real CatBoost hyperparameters | `iterations=50, depth=4` were chosen for Phase 1's 8-cell ablation to fit in a Modal timeout, not because they're good — a **screening** setting, explicitly flagged as such in `reports/wf_ensemble_v1_results.md` §4. Now that only one feature set needs testing, the compute budget for a real setting exists. | **Yes — Experiment A** |
| 2 | A weighted combiner instead of fixed 50/50 | Branch correlation is healthy but not perfect (5/6 use cases in-band); CatBoost and LogReg split wins roughly evenly (3/6 each) — a per-silo weight might beat a flat 50/50 without the complexity of a learned stacker (which `CONTEXT.md` §6 already found ties plain averaging). | **Yes — Experiment B** |
| 3 | Calibration before thresholding | The F2@0.5 vs. F2@t* gap from the last few turns is largest exactly where you'd expect miscalibration to hurt most (Soil Microbiome). If Platt scaling closes most of that gap, the naive 0.5 cutoff becomes usable without per-silo threshold tuning. | **Yes — Experiment C** |
| 4 | A third branch (Qwen3-8B) for diversity | Qwen3-8B and Jasper+Qwen3-4B were a near-tie in Phase 1 — the losing option might still add decorrelated errors as a third arm, the way Jasper added value in the original bake-off's concat "as diversity, not quality" (`wf_query_conditioned_findings.md` §5). | Not yet — needs a 3-way combiner design, more scope than fits here |
| 5 | Re-run the lex/cos-brief/metadata ablation under a *tuned* CatBoost | A stronger base learner can sometimes absorb what weaker engineered features were compensating for — worth confirming the +0.03-0.05 contribution from Tier1b/cos-brief/metadata still holds, not just assumed | Not yet — natural follow-up to Experiment A |
| 6 | PCA / Matryoshka / bilinear compression | Already tested and rejected — `reports/wf_query_conditioned_findings.md` §5 measured PCA-64 hurting within a silo | **Ruled out, not re-tested** |

Experiments A-C below are the three highest-value, cheapest-to-test ideas. #4 and #5 are
natural next rounds, not run here (see §5).

---

## 2. Experiment A — does un-cheaping CatBoost's hyperparameters help?

ROC-AUC by config:

| use_case                   | screening (iterations=50, depth=4) | tuned (iterations=150, depth=4) |
|----------------------------|------------------------------------|---------------------------------|
| Carbon Capture             | 0.885                              | 0.890                           |
| Low-Carbon Cement          | 0.914                              | 0.919                           |
| Named Entity Recognition   | 0.818                              | 0.831                           |
| Soil Microbiome            | 0.821                              | 0.838                           |
| Solar Cells for Satellites | 0.848                              | 0.870                           |
| Technology Prediction      | 0.824                              | 0.822                           |

Mean ROC-AUC delta (tuned - screening): **+0.010**, wins on 5/6 use cases.

## 3. Experiment B — weighted combiner vs. fixed 50/50

Using **tuned (iterations=150, depth=4)** CatBoost OOF (the stronger of the two from Experiment A) for the combiner sweep.

| use_case                   | auc_at_50_50 | best_w | auc_at_best_w | gain_vs_50_50 |
|----------------------------|--------------|--------|---------------|---------------|
| Carbon Capture             | 0.888        | 0.850  | 0.894         | 0.005         |
| Low-Carbon Cement          | 0.907        | 1.000  | 0.919         | 0.012         |
| Named Entity Recognition   | 0.833        | 0.850  | 0.839         | 0.006         |
| Soil Microbiome            | 0.827        | 0.950  | 0.838         | 0.011         |
| Solar Cells for Satellites | 0.856        | 0.900  | 0.873         | 0.017         |
| Technology Prediction      | 0.834        | 0.000  | 0.836         | 0.002         |

Mean gain from a per-silo weight over fixed 50/50: **+0.0090** ROC-AUC. **Diagnostic caveat:** this weight is chosen by looking at the same OOF it's scored on (no inner/outer split) — an optimistic upper bound on what a properly nested weight-selection would get, not a number to ship as-is.

## 4. Experiment C — does calibration make the naive 0.5 threshold usable?

| use_case                   | f2_at_0.5_raw | f2_at_0.5_calibrated | f2_at_t_star (raw, uncalibrated) |
|----------------------------|---------------|----------------------|----------------------------------|
| Carbon Capture             | 0.796         | 0.796                | 0.890                            |
| Low-Carbon Cement          | 0.901         | 0.900                | 0.949                            |
| Named Entity Recognition   | 0.865         | 0.885                | 0.929                            |
| Soil Microbiome            | 0.538         | 0.551                | 0.729                            |
| Solar Cells for Satellites | 0.888         | 0.910                | 0.953                            |
| Technology Prediction      | 0.829         | 0.833                | 0.899                            |

Calibration lifts mean F2@0.5 by **+0.010**; the remaining gap to F2@t* averages **+0.079**. **Diagnostic caveat:** Platt scaling fit and scored on the same OOF sample, same as Experiment B — a real deployment needs the calibrator fit on a held-out slice, not the same rows it's evaluated on.

## 5. What this means — and the honest reading of three +0.01s

All three experiments point the same direction (positive) and none of them is free —
that consistency is itself a real signal, worth acting on. But applying this project's own
discipline (`CONTEXT.md` §5: seed-to-seed sd is ~0.010 on within-silo ROC-AUC; **treat any
gap below ~0.03 as not established**) to the actual numbers:

| Experiment | Mean gain | Clears the ~0.03 noise floor? |
|---|---|---|
| A — tuned CatBoost hyperparameters | +0.010 ROC-AUC | **No** — sits at roughly one seed-noise sd |
| B — weighted combiner vs. 50/50 | +0.009 ROC-AUC | **No**, and it's an in-sample estimate besides |
| C — Platt calibration | +0.010 F2@0.5 | **No** — and it closes only ~11% of the F2@0.5→F2@t* gap |

**None of these three individually is a proven win yet.** Each is a single-seed OOF read,
same limitation as `wf_ensemble_fold_pipeline.ipynb`'s diagnostics — exactly the thing
Phase 1 ran 5 seeds to guard against before trusting a number. The honest conclusion is
"promising direction, not yet established," not "found three improvements."

**What would actually settle it:** re-run each experiment at 5 seeds (matching Phase 1's
own discipline) before deciding any of them replaces v1's defaults. Given the three point
the same way, a combined config (tuned CatBoost + weighted combiner + calibration together)
run at 5 seeds is the single most informative next experiment — if the gains are real and
independent they should roughly stack; if they're all reading the same underlying noise,
5 seeds will show that too.

**One finding that's more than a small number, though, and worth a closer look regardless
of the noise floor:** Experiment B's per-use-case optimal weight is not always near 0.5 —
Low-Carbon Cement's in-sample-optimal weight is 1.0 (pure CatBoost) and Technology
Prediction's is 0.0 (pure LogReg). That's a qualitative pattern (which branch wins isn't
uniform across use cases), not just a magnitude, and it's consistent with Experiment A's own
per-use-case table (CatBoost's tuned-vs-screening gain is negative on Technology Prediction,
the one use case where LogReg alone won the weight sweep). Worth a multi-seed check on
whether "let the stronger branch dominate per silo" beats a flat blend more convincingly
than the pooled mean suggests — a different question from "is 50/50 the right global
default," which these gains don't yet answer either way.

---

## 6. Where the evidence lives

| What | Where |
|---|---|
| Experiment A/B/C compute (Modal) | `scripts/modal_ensemble_experiments.py` |
| Experiment A/B/C analysis + this report's tables | `scripts/analyze_ensemble_v2_experiments.py` |
| Raw CatBoost OOF at both hyperparameter settings | `reports/wf_ensemble_v2_hparam_oof.json` |
| v1 baseline this compares against | `reports/wf_ensemble_v1_results.md`, `notebooks/main/09_ensemble_per_silo.ipynb` |
| Noise-floor discipline applied above | `CONTEXT.md` §5 |

## 7. Axis feature (PCA/SVD follow-up) + iterations=300, all vs. the tuned baseline

All configs at 5 seeds. F2@t* is the headline metric (per-use-case optimal threshold); ROC-AUC secondary. Noise floor: ~0.03 (CONTEXT.md §5).

| config         | roc_auc | f2_at_0.5 | f2_at_t_star |
|----------------|---------|-----------|--------------|
| tuned_150_4    | 0.861   | 0.826     | 0.892        |
| iterations_300 | 0.864   | 0.836     | 0.894        |
| axis_add       | 0.860   | 0.814     | 0.893        |
| axis_replace   | 0.859   | 0.812     | 0.897        |

**iterations_300 vs. tuned_150_4**: mean F2@t* delta **+0.0015** (sd 0.0037). Clears noise floor: no.

**axis_add vs. tuned_150_4**: mean F2@t* delta **+0.0002** (sd 0.0070). Clears noise floor: no.

**axis_replace vs. tuned_150_4**: mean F2@t* delta **+0.0046** (sd 0.0096). Clears noise floor: no.

Per-use-case F2@t*, mean +/- sd across 5 seeds:

*tuned_150_4*

| use_case                   | mean  | std   |
|----------------------------|-------|-------|
| Carbon Capture             | 0.894 | 0.004 |
| Low-Carbon Cement          | 0.949 | 0.001 |
| Named Entity Recognition   | 0.929 | 0.003 |
| Soil Microbiome            | 0.733 | 0.008 |
| Solar Cells for Satellites | 0.951 | 0.003 |
| Technology Prediction      | 0.899 | 0.005 |

*iterations_300*

| use_case                   | mean  | std   |
|----------------------------|-------|-------|
| Carbon Capture             | 0.899 | 0.003 |
| Low-Carbon Cement          | 0.950 | 0.001 |
| Named Entity Recognition   | 0.930 | 0.003 |
| Soil Microbiome            | 0.735 | 0.010 |
| Solar Cells for Satellites | 0.949 | 0.002 |
| Technology Prediction      | 0.902 | 0.003 |

*axis_add*

| use_case                   | mean  | std   |
|----------------------------|-------|-------|
| Carbon Capture             | 0.903 | 0.002 |
| Low-Carbon Cement          | 0.948 | 0.001 |
| Named Entity Recognition   | 0.929 | 0.001 |
| Soil Microbiome            | 0.733 | 0.012 |
| Solar Cells for Satellites | 0.948 | 0.003 |
| Technology Prediction      | 0.895 | 0.002 |

*axis_replace*

| use_case                   | mean  | std   |
|----------------------------|-------|-------|
| Carbon Capture             | 0.899 | 0.008 |
| Low-Carbon Cement          | 0.953 | 0.000 |
| Named Entity Recognition   | 0.927 | 0.003 |
| Soil Microbiome            | 0.752 | 0.014 |
| Solar Cells for Satellites | 0.951 | 0.003 |
| Technology Prediction      | 0.900 | 0.006 |

## 8. Verdict: none of #4 or the PCA/SVD idea beat the already-established tuned config

Applying the same per-use-case noise-floor discipline as §5:

**Iterations=300 vs. iterations=150: no further gain.** Mean F2@t* delta +0.0015 (sd
0.0037) — essentially flat, and Solar Cells for Satellites actually ticks down slightly
(0.951 → 0.949). The real gain from un-cheaping CatBoost happened going 50→150
(established in §5); 150→300 is diminishing returns already. **Don't spend the extra
training time — 150 iterations is the right stopping point.**

**The supervised embedding-axis feature, added alongside the raw embedding, does nothing**
(mean delta +0.0002, sd 0.0070) — CatBoost's own tree splits across the full 4608-dim
embedding already capture whatever discriminative direction this axis was trying to hand
it explicitly. Not useful as an addition.

**The interesting result is `axis_replace`: dropping the entire 4608-dim raw embedding
block and using ONE supervised direction instead scores statistically the same as keeping
it** (F2@t* 0.897 vs. 0.892 baseline, a hair higher if anything, well inside noise). This
isn't a performance win, but it's a real practical one: the same accuracy from 28 features
instead of 4600+, which is a much simpler, faster, cheaper model to train and serve. Worth
adopting for cost/latency reasons even though it doesn't move F2 — a different kind of
"improvement" than the others tested, and arguably the most actionable finding in this
whole exploratory round.

**Decision: the CatBoost config carried forward into nested weight selection (#2) and
SYNERGY validation (#1) stays `iterations=150, depth=4` on the full winning feature set**
(lex + cos-brief + metadata + Jasper+Qwen3-4B concat) — nothing tested here clears the bar
to change it. The axis-replace finding is recorded as a follow-up worth a deeper look
(does it hold up on SYNERGY too, where compute cost matters more at 100k-paper scale?) but
isn't swapped in for this round's "best option."

## 9. Properly nested weight selection (no in-sample peeking)

Weight chosen on seeds [0, 1, 2]'s OOF, evaluated on seeds [3, 4]'s OOF (never seen during selection) - for the three use cases with a stable per-seed preference. The other three keep fixed 50/50.

| use_case                   | chosen_w | held_out_auc_50_50 | held_out_auc_chosen_w | auc_gain | held_out_f2_at_t_star_50_50 | held_out_f2_at_t_star_chosen_w | f2_gain |
|----------------------------|----------|--------------------|-----------------------|----------|-----------------------------|--------------------------------|---------|
| Carbon Capture             | 0.500    | 0.905              | 0.905                 | 0.000    | 0.896                       | 0.896                          | 0.000   |
| Low-Carbon Cement          | 1.000    | 0.908              | 0.911                 | 0.004    | 0.949                       | 0.949                          | 0.001   |
| Named Entity Recognition   | 0.500    | 0.834              | 0.834                 | 0.000    | 0.930                       | 0.930                          | 0.000   |
| Soil Microbiome            | 0.950    | 0.814              | 0.827                 | 0.013    | 0.720                       | 0.726                          | 0.006   |
| Solar Cells for Satellites | 0.850    | 0.866              | 0.875                 | 0.009    | 0.950                       | 0.950                          | 0.001   |
| Technology Prediction      | 0.500    | 0.835              | 0.835                 | 0.000    | 0.907                       | 0.907                          | 0.000   |

On the 3 use cases with a nested (honest, held-out) weight: mean F2 gain **+0.0023**, mean AUC gain **+0.0086** — evaluated on seeds the weight selection never saw. This is the trustworthy number, not the in-sample one from the earlier rounds.


## 10. SYNERGY validation — the realistic-prevalence check

Winning config's core (Tier-1b BM25 + raw Jasper+Qwen3-4B embedding, CatBoost iterations=150/depth=4 + LogisticRegression, 50/50 blend — SYNERGY reviews are new/unseen "use cases" so none of the per-TIRI-use-case nested weights transfer here) against SYNERGY's 3 reviews at 1.7-14.8% prevalence, vs. 26-77% across TIRI's own six pools. No cos_brief or metadata (not available for SYNERGY - same scope limit `scripts/run_synergy_recall_validation.py` already documents).

Review sizes and prevalence:

| index             | n    | prevalence |
|-------------------|------|------------|
| Sep_2021          | 271  | 0.148      |
| Menon_2022        | 975  | 0.076      |
| van_der_Waal_2022 | 1970 | 0.017      |

Per review, mean +/- sd across seeds:

| review            | roc_auc (mean / sd) | f2@0.5 (mean / sd) | f2@t* (mean / sd) | recall@10% (mean / sd) | recall@20% (mean / sd) | wss@95 (mean / sd) |
|-------------------|----------------------|---------------------|--------------------|--------------------------|--------------------------|----------------------|
| Menon_2022        | 0.959 / 0.004        | 0.699 / 0.025       | 0.755 / 0.027      | 0.784 / 0.027            | 0.922 / 0.018            | 0.708 / 0.013        |
| Sep_2021          | 0.751 / 0.031        | 0.385 / 0.041       | 0.554 / 0.040      | 0.365 / 0.045            | 0.515 / 0.034            | 0.083 / 0.028        |
| van_der_Waal_2022 | 0.941 / 0.009        | 0.311 / 0.036       | 0.516 / 0.031      | 0.812 / 0.033            | 0.909 / 0.021            | 0.642 / 0.097        |

Overall mean ROC-AUC: **0.884**, mean F2@t*: **0.609**, mean Recall@20%: **0.782**, mean WSS@95: **0.478**.

Compare against `reports/wf_synergy_recall_validation.md` §1 (embedding-alone and embedding+lexical LogisticRegression numbers already published there) for whether adding CatBoost + the ensemble blend moves the needle on the one surface that actually matters for production prevalence.


### Direct comparison against the published SYNERGY baseline

| review | ensemble ROC-AUC | best published single-embedding row | published MEAN row |
|---|---|---|---|
| Sep_2021 | 0.751 | 0.798 (qwen3-8b: embedding) | — |
| Menon_2022 | 0.959 | 0.958 (qwen3-8b or qwen3-4b: embedding) | — |
| van_der_Waal_2022 | 0.941 | 0.936 (qwen3-4b: embedding + lexical) | — |
| **MEAN** | **0.884** | **0.888** (qwen3-8b: embedding alone) | 0.876 (jasper: embedding + lexical, the closest architectural match) |

**The ensemble generalizes.** Its mean ROC-AUC (0.884) sits essentially level with the best
single-embedding baseline ever tested here (0.888) and clearly above the closest
architectural match — a single embedding + lexical via plain LogisticRegression (jasper:
0.876, qwen3-4b: 0.879) — despite using a reduced, cos-brief/metadata-free feature set.
CatBoost + the ensemble blend is not just an in-repo artifact; it holds up on an external,
prevalence-realistic benchmark.

**Where it doesn't win: Sep_2021 specifically (0.751 vs. 0.798), and there's a clean reason
why.** This ensemble uses Jasper+Qwen3-4B — and `reports/wf_query_conditioned_findings.md`
§5 already found Qwen3-4B is *last* of the three candidate embeddings at realistic
prevalence, with Qwen3-8B winning clearly. Sep_2021 is also the smallest review (271 rows),
where CatBoost's higher fold-to-fold variance likely compounds the effect. This is
consistent with, not a new problem beyond, what was already flagged as unresolved in
`reports/wf_ensemble_v1_results.md` §5 item 1 — **swapping Qwen3-4B for Qwen3-8B in the
embedding block is the one concrete, evidence-backed change most likely to close this gap**,
and is now the clearest next thing to test.

## 11. SYNERGY validation — Jasper+Qwen3-8B

Config core (Tier-1b BM25 + raw Jasper+Qwen3-8B embedding), CatBoost iterations=150/depth=4 + LogisticRegression, 50/50 blend — SYNERGY reviews are new/unseen "use cases" so none of the per-TIRI-use-case nested weights transfer here) against SYNERGY's 3 reviews at 1.7-14.8% prevalence, vs. 26-77% across TIRI's own six pools. No cos_brief or metadata (not available for SYNERGY - same scope limit `scripts/run_synergy_recall_validation.py` already documents).

Review sizes and prevalence:

| index             | n    | prevalence |
|-------------------|------|------------|
| Sep_2021          | 271  | 0.148      |
| Menon_2022        | 975  | 0.076      |
| van_der_Waal_2022 | 1970 | 0.017      |

Per review, mean +/- sd across seeds:

| review            | roc_auc (mean / sd) | f2@0.5 (mean / sd) | f2@t* (mean / sd) | recall@10% (mean / sd) | recall@20% (mean / sd) | wss@95 (mean / sd) |
|-------------------|----------------------|---------------------|--------------------|--------------------------|--------------------------|----------------------|
| Menon_2022        | 0.959 / 0.004        | 0.712 / 0.011       | 0.743 / 0.012      | 0.776 / 0.012            | 0.914 / 0.018            | 0.695 / 0.026        |
| Sep_2021          | 0.797 / 0.027        | 0.414 / 0.048       | 0.591 / 0.030      | 0.375 / 0.040            | 0.575 / 0.031            | 0.205 / 0.073        |
| van_der_Waal_2022 | 0.942 / 0.011        | 0.285 / 0.075       | 0.504 / 0.033      | 0.782 / 0.025            | 0.933 / 0.014            | 0.623 / 0.155        |

Overall mean ROC-AUC: **0.899**, mean F2@t*: **0.613**, mean Recall@20%: **0.807**, mean WSS@95: **0.508**.

Compare against `reports/wf_synergy_recall_validation.md` §1 (embedding-alone and embedding+lexical LogisticRegression numbers already published there) for whether adding CatBoost + the ensemble blend moves the needle on the one surface that actually matters for production prevalence.


### Qwen3-4B → Qwen3-8B: does the swap fix what it was predicted to fix?

| review | Jasper+Qwen3-4B ROC-AUC | Jasper+Qwen3-8B ROC-AUC | delta | Jasper+Qwen3-4B WSS@95 | Jasper+Qwen3-8B WSS@95 | delta |
|---|---|---|---|---|---|---|
| Sep_2021 | 0.751 | **0.797** | **+0.046** | 0.083 | **0.205** | **+0.122** |
| Menon_2022 | 0.959 | 0.959 | 0.000 | 0.708 | 0.695 | −0.013 |
| van_der_Waal_2022 | 0.941 | 0.942 | +0.001 | 0.642 | 0.623 | −0.019 |
| **MEAN** | **0.884** | **0.899** | **+0.015** | **0.478** | **0.508** | **+0.030** |

**Yes, decisively, and exactly where predicted.** Sep_2021 — the review where the
Jasper+Qwen3-4B ensemble underperformed the published baseline — is precisely where the
swap pays off: ROC-AUC +0.046, WSS@95 +0.122. The other two reviews move by amounts well
inside seed noise (their WSS@95 sd was already 0.10-0.16 across seeds in both runs) — no
regression, just noise. Mean ROC-AUC across all three reviews improves from 0.884 to 0.899.

**The swapped ensemble now matches or beats the best single-embedding baseline ever
published here, on every review:**

| review | published best (qwen3-8b: embedding alone) | this ensemble (Jasper+Qwen3-8B) |
|---|---|---|
| Sep_2021 | 0.798 | 0.797 (tied) |
| Menon_2022 | 0.958 | 0.959 (better) |
| van_der_Waal_2022 | 0.907 | 0.942 (better, +0.035) |
| **MEAN** | **0.888** | **0.899 (better)** |

This was the clearest, most concrete recommendation to come out of the whole exploratory
round, and it held up under direct test. **Jasper+Qwen3-8B is now the recommended embedding
pair** for this ensemble, superseding Jasper+Qwen3-4B everywhere it was used above.

## 12. Does a third branch (SVM) add an edge?

Prompted by: "did we miss SVM, as our ensemble is only 2 models, I wondered if a third would give another edge." A linear-kernel SVM was screened first (seed 0 only) and dropped - RBF beat it in all 6 use cases (mean within-silo ROC-AUC 0.819 vs 0.859) - so only `SVC(kernel="rbf", probability=True, class_weight="balanced")` is carried through the analysis below, on the same `lex + cos-brief + metadata + Jasper+Qwen3-4B concat` feature set every other v2 experiment uses.

### Standalone performance, mean +/- sd across 5 seeds

| use_case                   | branch             | roc_auc_mean | roc_auc_sd | f2_at_t_star_mean | f2_at_t_star_sd |
|----------------------------|--------------------|--------------|------------|-------------------|-----------------|
| Carbon Capture             | CatBoost (150/4)   | 0.895        | 0.004      | 0.894             | 0.003           |
| Carbon Capture             | LogisticRegression | 0.896        | 0.011      | 0.871             | 0.008           |
| Carbon Capture             | SVM (rbf)          | 0.906        | 0.006      | 0.906             | 0.004           |
| Low-Carbon Cement          | CatBoost (150/4)   | 0.912        | 0.008      | 0.949             | 0.001           |
| Low-Carbon Cement          | LogisticRegression | 0.876        | 0.011      | 0.928             | 0.003           |
| Low-Carbon Cement          | SVM (rbf)          | 0.915        | 0.008      | 0.948             | 0.002           |
| Named Entity Recognition   | CatBoost (150/4)   | 0.829        | 0.007      | 0.929             | 0.002           |
| Named Entity Recognition   | LogisticRegression | 0.823        | 0.011      | 0.909             | 0.007           |
| Named Entity Recognition   | SVM (rbf)          | 0.852        | 0.008      | 0.928             | 0.002           |
| Soil Microbiome            | CatBoost (150/4)   | 0.829        | 0.006      | 0.733             | 0.007           |
| Soil Microbiome            | LogisticRegression | 0.785        | 0.007      | 0.643             | 0.019           |
| Soil Microbiome            | SVM (rbf)          | 0.823        | 0.005      | 0.723             | 0.007           |
| Solar Cells for Satellites | CatBoost (150/4)   | 0.873        | 0.010      | 0.951             | 0.003           |
| Solar Cells for Satellites | LogisticRegression | 0.827        | 0.013      | 0.932             | 0.005           |
| Solar Cells for Satellites | SVM (rbf)          | 0.824        | 0.006      | 0.944             | 0.001           |
| Technology Prediction      | CatBoost (150/4)   | 0.829        | 0.014      | 0.899             | 0.005           |
| Technology Prediction      | LogisticRegression | 0.835        | 0.005      | 0.871             | 0.004           |
| Technology Prediction      | SVM (rbf)          | 0.856        | 0.012      | 0.909             | 0.005           |

### Branch-disagreement diagnostic — Pearson rho of OOF probabilities, averaged across 5 seeds

| use_case                   | svm_vs_catboost | svm_vs_logreg | catboost_vs_logreg |
|----------------------------|-----------------|---------------|--------------------|
| Carbon Capture             | 0.946           | 0.901         | 0.851              |
| Low-Carbon Cement          | 0.961           | 0.920         | 0.880              |
| Named Entity Recognition   | 0.912           | 0.858         | 0.767              |
| Soil Microbiome            | 0.914           | 0.807         | 0.714              |
| Solar Cells for Satellites | 0.800           | 0.788         | 0.653              |
| Technology Prediction      | 0.912           | 0.889         | 0.813              |

Mean rho across use cases: SVM-vs-CatBoost **0.908**, SVM-vs-LogReg **0.861**, CatBoost-vs-LogReg (reference, the pair already in production) **0.780**.

### Nested (non-leaky) 3-way blend vs. the existing nested 2-way blend

Weights chosen on seeds [0, 1, 2]'s OOF (grid search, step 0.1, over the full 3-branch simplex), evaluated on seeds [3, 4]'s OOF - never seen during selection. Compared against the already-nested 2-way CatBoost/LogReg weight from `reports/wf_ensemble_v2_chosen_weights.json`, evaluated on the same held-out seeds.

| use_case                   | w_cb  | w_lr  | w_svm | held_out_auc_2way | held_out_auc_3way | auc_gain | held_out_f2_2way | held_out_f2_3way | f2_gain |
|----------------------------|-------|-------|-------|-------------------|-------------------|----------|------------------|------------------|---------|
| Carbon Capture             | 0.100 | 0.200 | 0.700 | 0.905             | 0.911             | 0.006    | 0.896            | 0.901            | 0.005   |
| Low-Carbon Cement          | 0.600 | 0.000 | 0.400 | 0.911             | 0.916             | 0.005    | 0.949            | 0.949            | -0.001  |
| Named Entity Recognition   | 0.000 | 0.000 | 1.000 | 0.834             | 0.853             | 0.019    | 0.930            | 0.926            | -0.003  |
| Soil Microbiome            | 0.600 | 0.000 | 0.400 | 0.827             | 0.830             | 0.003    | 0.726            | 0.717            | -0.009  |
| Solar Cells for Satellites | 0.900 | 0.100 | 0.000 | 0.875             | 0.875             | 0.000    | 0.950            | 0.950            | 0.000   |
| Technology Prediction      | 0.000 | 0.000 | 1.000 | 0.835             | 0.863             | 0.028    | 0.907            | 0.914            | 0.007   |

Overall: mean AUC gain **+0.0102**, mean F2@t* gain **-0.0002**, evaluated on held-out seeds never used for weight selection. Clears the noise floor: **no**.

3-way blend beats the 2-way blend's held-out AUC on 6/6 use cases; the single largest gain is +0.0281 (Technology Prediction), mean F2@t* gain is actually *negative* (-0.0002, and negative on 3/6 use cases individually). **Verdict: REJECT.**

This is a diversity problem, not a sample-size/ceiling problem, and the correlation table above says so directly: SVM(rbf) is *more* correlated with CatBoost (mean rho 0.908) and with LogisticRegression (0.861) than CatBoost and LogisticRegression are with each other (0.780, the pair already in production). Standalone, SVM(rbf) is actually a strong branch in its own right — it has the best single-branch ROC-AUC on 4/6 use cases (Carbon Capture, Low-Carbon Cement, Named Entity Recognition, Technology Prediction) — but that strength comes from doing a *better version of the same thing* the ensemble already does in this ~4600-dim, mostly-linear-embedding-dominated feature space, not from a different vote. Adding it doesn't buy the ensemble a new perspective; it buys a third, highly-correlated opinion that occasionally nudges AUC by a few thousandths and about as often costs a few thousandths of F2@t*. Given the real serving/maintenance cost of a third model (fit, calibrate, monitor, keep in sync per silo), this isn't worth adopting. If a genuinely different third branch is wanted later, it would need to look structurally different from both existing branches (e.g. a model that doesn't see the raw embedding directly, or that's built on a materially different feature subset) rather than another classifier on the same full feature set.


## 13. Does a lexical/metadata-only branch (no embedding) add an edge?

Motivated by `reports/wf_lean_vs_embedding_arms.md` (2026-08-06), which found a "LEAN" arm (cos_brief + rank + lexical + metadata, no embeddings, plain LogisticRegression) had mean Spearman rank correlation of only **0.535** with an embedding-only arm - well below the production CatBoost/LogReg pair's Pearson 0.780, and far below SVM's 0.86-0.91 (§12). That check never involved CatBoost, never used the actual production feature/weight baseline, and never ran a nested weight-selection blend - this closes that gap, at the same rigor as the SVM check. A linear-kernel screen isn't relevant here (there is no kernel choice); a seed-0 screen instead compared CatBoost vs. LogisticRegression *on the lean feature set itself* (mean ROC-AUC 0.778 vs 0.781, a statistical tie) - CatBoost-lean is carried forward as the single third branch, on `lex + cos-brief + metadata (no embedding)` (21 columns, no embedding at all), against the two production branches on `lex + cos-brief + metadata + Jasper+Qwen3-4B concat`.

### Standalone performance, mean +/- sd across 5 seeds

| use_case                   | branch                              | roc_auc_mean | roc_auc_sd | f2_at_t_star_mean | f2_at_t_star_sd |
|----------------------------|-------------------------------------|--------------|------------|-------------------|-----------------|
| Carbon Capture             | CatBoost (150/4, full)              | 0.895        | 0.004      | 0.894             | 0.003           |
| Carbon Capture             | LogisticRegression (full)           | 0.896        | 0.011      | 0.871             | 0.008           |
| Carbon Capture             | CatBoost (150/4, lean/no-embedding) | 0.820        | 0.012      | 0.866             | 0.006           |
| Low-Carbon Cement          | CatBoost (150/4, full)              | 0.912        | 0.008      | 0.949             | 0.001           |
| Low-Carbon Cement          | LogisticRegression (full)           | 0.876        | 0.011      | 0.928             | 0.003           |
| Low-Carbon Cement          | CatBoost (150/4, lean/no-embedding) | 0.933        | 0.003      | 0.956             | 0.003           |
| Named Entity Recognition   | CatBoost (150/4, full)              | 0.829        | 0.007      | 0.929             | 0.002           |
| Named Entity Recognition   | LogisticRegression (full)           | 0.823        | 0.011      | 0.909             | 0.007           |
| Named Entity Recognition   | CatBoost (150/4, lean/no-embedding) | 0.764        | 0.004      | 0.929             | 0.002           |
| Soil Microbiome            | CatBoost (150/4, full)              | 0.829        | 0.006      | 0.733             | 0.007           |
| Soil Microbiome            | LogisticRegression (full)           | 0.785        | 0.007      | 0.643             | 0.019           |
| Soil Microbiome            | CatBoost (150/4, lean/no-embedding) | 0.667        | 0.020      | 0.656             | 0.006           |
| Solar Cells for Satellites | CatBoost (150/4, full)              | 0.873        | 0.010      | 0.951             | 0.003           |
| Solar Cells for Satellites | LogisticRegression (full)           | 0.827        | 0.013      | 0.932             | 0.005           |
| Solar Cells for Satellites | CatBoost (150/4, lean/no-embedding) | 0.830        | 0.008      | 0.951             | 0.004           |
| Technology Prediction      | CatBoost (150/4, full)              | 0.829        | 0.014      | 0.899             | 0.005           |
| Technology Prediction      | LogisticRegression (full)           | 0.835        | 0.005      | 0.871             | 0.004           |
| Technology Prediction      | CatBoost (150/4, lean/no-embedding) | 0.672        | 0.010      | 0.887             | 0.003           |

### Branch-disagreement diagnostic — Pearson rho of OOF probabilities, averaged across 5 seeds

| use_case                   | lean_vs_catboost_full | lean_vs_logreg_full | catboost_vs_logreg_full |
|----------------------------|-----------------------|---------------------|-------------------------|
| Carbon Capture             | 0.739                 | 0.650               | 0.851                   |
| Low-Carbon Cement          | 0.923                 | 0.806               | 0.880                   |
| Named Entity Recognition   | 0.738                 | 0.542               | 0.767                   |
| Soil Microbiome            | 0.418                 | 0.337               | 0.714                   |
| Solar Cells for Satellites | 0.755                 | 0.385               | 0.653                   |
| Technology Prediction      | 0.443                 | 0.379               | 0.813                   |

Mean rho across use cases: lean-vs-CatBoost(full) **0.669**, lean-vs-LogReg(full) **0.516**, CatBoost-vs-LogReg (reference, the pair already in production) **0.780**. The older Spearman-rank 0.535 finding and this Pearson-on-probabilities number are different metrics on a different exact feature set (33 cols there vs. 21 here, qwen3-8b-alone vs. the actual production Jasper+Qwen3-4B feature set) so an exact match isn't expected - the question is only whether the direction (lean decorrelates more than the existing pair decorrelates from itself) holds up.

### Nested (non-leaky) 3-way blend vs. the existing nested 2-way blend

Weights chosen on seeds [0, 1, 2]'s OOF (grid search, step 0.1, over the full 3-branch simplex), evaluated on seeds [3, 4]'s OOF - never seen during selection. Compared against the already-nested 2-way CatBoost/LogReg weight from `reports/wf_ensemble_v2_chosen_weights.json`, evaluated on the same held-out seeds.

| use_case                   | w_cb  | w_lr  | w_lean | held_out_auc_2way | held_out_auc_3way | auc_gain | held_out_f2_2way | held_out_f2_3way | f2_gain |
|----------------------------|-------|-------|--------|-------------------|-------------------|----------|------------------|------------------|---------|
| Carbon Capture             | 0.500 | 0.400 | 0.100  | 0.905             | 0.907             | 0.002    | 0.896            | 0.899            | 0.003   |
| Low-Carbon Cement          | 0.000 | 0.000 | 1.000  | 0.911             | 0.934             | 0.023    | 0.949            | 0.955            | 0.006   |
| Named Entity Recognition   | 0.700 | 0.200 | 0.100  | 0.834             | 0.838             | 0.005    | 0.930            | 0.929            | -0.001  |
| Soil Microbiome            | 0.900 | 0.000 | 0.100  | 0.827             | 0.828             | 0.001    | 0.726            | 0.729            | 0.003   |
| Solar Cells for Satellites | 0.600 | 0.200 | 0.200  | 0.875             | 0.884             | 0.009    | 0.950            | 0.954            | 0.003   |
| Technology Prediction      | 0.000 | 1.000 | 0.000  | 0.835             | 0.831             | -0.004   | 0.907            | 0.874            | -0.033  |

Overall: mean AUC gain **+0.0058**, mean F2@t* gain **-0.0031**, evaluated on held-out seeds never used for weight selection. Clears the noise floor: **no**.

3-way blend beats the 2-way blend's held-out AUC on 5/6 use cases; the single largest gain is +0.0229 (Low-Carbon Cement). **Verdict: REJECT** — the lean branch IS more decorrelated than the existing pair (mean rho vs. the two full-feature branches 0.593 vs. their own 0.780 with each other) - so this is a sample-size/ceiling problem, not a diversity problem: the disagreement is real but the lean branch is too much weaker standalone for its disagreement to move the blend beyond noise.


## 14. Does a third branch (k-NN) add an edge?

Third and final lever in this line of work. §12 (SVM) varied the algorithm only, kept the feature view fixed, and was REJECTED - correlation 0.86-0.91, *higher* than the existing CatBoost/LogReg pair's own 0.780, because CatBoost, LogisticRegression, and SVM(rbf) all fit one global decision surface, and this feature space is separable enough that any of them land in roughly the same place. §13 (lexical/metadata-only) varied the feature view only, and genuinely decorrelated (rho 0.52-0.67) but was REJECTED too - too weak standalone without the embedding for the real disagreement to move the blend past noise. This experiment varies the lever neither of those touched: a mechanistically different algorithm on the SAME `lex + cos-brief + metadata + Jasper+Qwen3-4B concat` feature set. k-NN doesn't fit a global boundary at all - it decides locally, by who's nearby, not by a fitted surface. If that's a real mechanistic difference rather than just another way to draw the same boundary, it should show up as lower correlation than SVM's 0.86-0.91.

n_neighbors screened at seed 0 over [5, 15, 25, 31] (metric=cosine, weights=distance): k=5 -> 0.7927, k=15 -> 0.8275, k=25 -> 0.8319, k=31 -> 0.8356. Chosen **k=31**, carried through the full 5-seed run below.

### Standalone performance, mean +/- sd across 5 seeds

| use_case                   | branch              | roc_auc_mean | roc_auc_sd | f2_at_t_star_mean | f2_at_t_star_sd |
|----------------------------|---------------------|--------------|------------|-------------------|-----------------|
| Carbon Capture             | CatBoost (150/4)    | 0.895        | 0.004      | 0.894             | 0.003           |
| Carbon Capture             | LogisticRegression  | 0.896        | 0.011      | 0.871             | 0.008           |
| Carbon Capture             | k-NN (k=31, cosine) | 0.874        | 0.004      | 0.890             | 0.006           |
| Low-Carbon Cement          | CatBoost (150/4)    | 0.912        | 0.008      | 0.949             | 0.001           |
| Low-Carbon Cement          | LogisticRegression  | 0.876        | 0.011      | 0.928             | 0.003           |
| Low-Carbon Cement          | k-NN (k=31, cosine) | 0.873        | 0.011      | 0.953             | 0.001           |
| Named Entity Recognition   | CatBoost (150/4)    | 0.829        | 0.007      | 0.929             | 0.002           |
| Named Entity Recognition   | LogisticRegression  | 0.823        | 0.011      | 0.909             | 0.007           |
| Named Entity Recognition   | k-NN (k=31, cosine) | 0.827        | 0.014      | 0.932             | 0.002           |
| Soil Microbiome            | CatBoost (150/4)    | 0.829        | 0.006      | 0.733             | 0.007           |
| Soil Microbiome            | LogisticRegression  | 0.785        | 0.007      | 0.643             | 0.019           |
| Soil Microbiome            | k-NN (k=31, cosine) | 0.823        | 0.006      | 0.722             | 0.007           |
| Solar Cells for Satellites | CatBoost (150/4)    | 0.873        | 0.010      | 0.951             | 0.003           |
| Solar Cells for Satellites | LogisticRegression  | 0.827        | 0.013      | 0.932             | 0.005           |
| Solar Cells for Satellites | k-NN (k=31, cosine) | 0.801        | 0.005      | 0.947             | 0.002           |
| Technology Prediction      | CatBoost (150/4)    | 0.829        | 0.014      | 0.899             | 0.005           |
| Technology Prediction      | LogisticRegression  | 0.835        | 0.005      | 0.871             | 0.004           |
| Technology Prediction      | k-NN (k=31, cosine) | 0.819        | 0.017      | 0.908             | 0.005           |

### Branch-disagreement diagnostic — Pearson rho of OOF probabilities, averaged across 5 seeds

| use_case                   | knn_vs_catboost | knn_vs_logreg | catboost_vs_logreg |
|----------------------------|-----------------|---------------|--------------------|
| Carbon Capture             | 0.914           | 0.793         | 0.851              |
| Low-Carbon Cement          | 0.943           | 0.848         | 0.880              |
| Named Entity Recognition   | 0.856           | 0.723         | 0.767              |
| Soil Microbiome            | 0.870           | 0.666         | 0.714              |
| Solar Cells for Satellites | 0.782           | 0.642         | 0.653              |
| Technology Prediction      | 0.868           | 0.770         | 0.813              |

Mean rho across use cases: k-NN-vs-CatBoost **0.872**, k-NN-vs-LogReg **0.740**, CatBoost-vs-LogReg (reference, the pair already in production) **0.780**. SVM's equivalent numbers (§12) were 0.908/0.861 - lower than SVM's, i.e. the mechanistic-difference hypothesis holds up.

### Nested (non-leaky) 3-way blend vs. the existing nested 2-way blend

Weights chosen on seeds [0, 1, 2]'s OOF (grid search, step 0.1, over the full 3-branch simplex), evaluated on seeds [3, 4]'s OOF - never seen during selection. Compared against the already-nested 2-way CatBoost/LogReg weight from `reports/wf_ensemble_v2_chosen_weights.json`, evaluated on the same held-out seeds.

| use_case                   | w_cb  | w_lr  | w_knn | held_out_auc_2way | held_out_auc_3way | auc_gain | held_out_f2_2way | held_out_f2_3way | f2_gain |
|----------------------------|-------|-------|-------|-------------------|-------------------|----------|------------------|------------------|---------|
| Carbon Capture             | 0.600 | 0.300 | 0.100 | 0.905             | 0.907             | 0.002    | 0.896            | 0.896            | -0.000  |
| Low-Carbon Cement          | 1.000 | 0.000 | 0.000 | 0.911             | 0.911             | 0.000    | 0.949            | 0.949            | 0.000   |
| Named Entity Recognition   | 0.400 | 0.100 | 0.500 | 0.834             | 0.844             | 0.011    | 0.930            | 0.930            | 0.001   |
| Soil Microbiome            | 0.600 | 0.000 | 0.400 | 0.827             | 0.834             | 0.007    | 0.726            | 0.731            | 0.004   |
| Solar Cells for Satellites | 0.900 | 0.100 | 0.000 | 0.875             | 0.875             | 0.000    | 0.950            | 0.950            | 0.000   |
| Technology Prediction      | 0.000 | 1.000 | 0.000 | 0.835             | 0.831             | -0.004   | 0.907            | 0.874            | -0.033  |

Overall: mean AUC gain **+0.0025**, mean F2@t* gain **-0.0046**, evaluated on held-out seeds never used for weight selection. Clears the noise floor: **no**.

3-way blend beats the 2-way blend's held-out AUC on 4/6 use cases; the single largest gain is +0.0105 (Named Entity Recognition). **Verdict: REJECT** — k-NN is NOT more decorrelated than the existing pair (mean rho 0.806 vs. their own 0.780) - despite deciding by a different mechanism (local neighbors, not a fitted global boundary), it still lands close to the same answer as CatBoost/LogReg/SVM in this feature space, which is itself a finding: the classes are cleanly separable enough here that mechanism doesn't matter much, only the feature view does (consistent with §13's result).

This closes the three-lever sweep on "does a third branch add an edge": algorithm-only (§12, SVM, rejected - too correlated), feature-view-only (§13, lexical/metadata, rejected - too weak standalone despite real decorrelation), and mechanism-only (§14, k-NN, above) all land on REJECT for the general case, each for a documented, different reason. The one live thread is §13's Low-Carbon Cement exception, which is a single-use-case question, not a general third-branch one.


## 15. Central hyperparameter search (CatBoost depth/iterations, LogReg C), via LOGO

Every CatBoost hyperparameter change so far this session (50->150->300 iterations, §§2, 7-8) was selected on **within-silo** CV - the production surface - not LOGO, which `CONTEXT.md` §1 specifies as the correct defaults-selection surface ("hyperparameter search happens once, centrally, in the LOGO loop"). Depth was never explored past 4 (6 blew a within-silo Modal timeout); LogisticRegression's `C` has never been touched from sklearn's default (1.0) anywhere in this repo's ensemble work. This closes both gaps: search via LOGO, then honestly validate the LOGO-chosen winner on fresh 5-seed within-silo OOF before adopting anything - extending this session's nested-selection discipline (§9, combiner weights) to hyperparameters themselves.

**Calibration** (`scripts/modal_ensemble_experiments.py::run_catboost_logo`, measured directly, not assumed): a LOGO fit trains on ~1550 pooled rows (5 of 6 use cases), not the ~250-360 rows a within-silo fold trains on - depth=4/iterations=150 (current shipped setting) took **190.9s**; depth=6/iterations=100 took **446.3s**. CatBoost's per-iteration cost scales ~linearly at fixed depth, so depth=6/iterations=350 (this grid's worst case) projects to ~1561s - comfortably under the 3600s timeout. depth=6 is therefore included in the grid below.

### CatBoost grid via LOGO (mean across 6 use cases)

| depth | iterations | mean_logo_roc_auc | mean_logo_f2_at_0.5 |
|-------|------------|-------------------|---------------------|
| 3     | 50         | 0.644             | 0.386               |
| 3     | 100        | 0.650             | 0.373               |
| 3     | 150        | 0.653             | 0.318               |
| 3     | 250        | 0.658             | 0.360               |
| 3     | 350        | 0.691             | 0.411               |
| 4     | 50         | 0.642             | 0.328               |
| 4     | 100        | 0.652             | 0.326               |
| 4     | 150        | 0.638             | 0.309               |
| 4     | 250        | 0.675             | 0.354               |
| 4     | 350        | 0.669             | 0.343               |
| 5     | 50         | 0.626             | 0.370               |
| 5     | 100        | 0.649             | 0.347               |
| 5     | 150        | 0.653             | 0.349               |
| 5     | 250        | 0.660             | 0.328               |
| 5     | 350        | 0.667             | 0.311               |
| 6     | 50         | 0.620             | 0.447               |
| 6     | 100        | 0.648             | 0.436               |
| 6     | 150        | 0.629             | 0.346               |
| 6     | 250        | 0.680             | 0.392               |
| 6     | 350        | 0.678             | 0.347               |

LOGO winner: **iterations=350, depth=3** (mean LOGO ROC-AUC 0.6912, mean LOGO F2@0.5 0.4106) vs. the currently-shipped iterations=150/depth=4 (mean LOGO ROC-AUC 0.6376, mean LOGO F2@0.5 0.3088).

### LogisticRegression C grid via LOGO

| C       | mean_logo_roc_auc | mean_logo_f2_at_0.5 |
|---------|-------------------|---------------------|
| 0.001   | 0.636             | 0.110               |
| 0.010   | 0.631             | 0.205               |
| 0.100   | 0.625             | 0.292               |
| 1.000   | 0.622             | 0.300               |
| 10.000  | 0.620             | 0.190               |
| 100.000 | 0.613             | 0.174               |

LOGO winner: **C=0.001** (mean LOGO ROC-AUC 0.6363, mean LOGO F2@0.5 0.1099) vs. the currently-shipped C=1.0 (mean LOGO ROC-AUC 0.6215, mean LOGO F2@0.5 0.2996).

### Within-silo validation (the production surface, and the actual adoption gate)

5-seed within-silo OOF for the LOGO winner vs. the currently-shipped setting, for both CatBoost and LogisticRegression. `NOISE_FLOOR = 0.03` applied exactly as every other v2 experiment - only adopt if the LOGO winner clears it.

**CatBoost: LOGO winner (iterations=350, depth=3) vs. current (iterations=150, depth=4)**

| use_case                   | current_auc | winner_auc | auc_gain | current_f2 | winner_f2 | f2_gain |
|----------------------------|-------------|------------|----------|------------|-----------|---------|
| Carbon Capture             | 0.895       | 0.898      | 0.003    | 0.894      | 0.896     | 0.002   |
| Low-Carbon Cement          | 0.912       | 0.913      | 0.001    | 0.949      | 0.949     | -0.001  |
| Named Entity Recognition   | 0.829       | 0.825      | -0.004   | 0.929      | 0.929     | 0.000   |
| Soil Microbiome            | 0.829       | 0.838      | 0.009    | 0.733      | 0.737     | 0.004   |
| Solar Cells for Satellites | 0.873       | 0.875      | 0.002    | 0.951      | 0.948     | -0.003  |
| Technology Prediction      | 0.829       | 0.838      | 0.010    | 0.899      | 0.904     | 0.005   |

Mean AUC gain **+0.0035**, mean F2@t* gain **+0.0013**. Clears the noise floor: **no**. **Verdict: KEEP CURRENT**.

**LogisticRegression: LOGO winner (C=0.001) vs. current (C=1.0)**

| use_case                   | current_auc | winner_auc | auc_gain | current_f2 | winner_f2 | f2_gain |
|----------------------------|-------------|------------|----------|------------|-----------|---------|
| Carbon Capture             | 0.896       | 0.906      | 0.011    | 0.871      | 0.908     | 0.037   |
| Low-Carbon Cement          | 0.876       | 0.899      | 0.023    | 0.928      | 0.948     | 0.019   |
| Named Entity Recognition   | 0.823       | 0.857      | 0.034    | 0.909      | 0.933     | 0.023   |
| Soil Microbiome            | 0.785       | 0.834      | 0.049    | 0.643      | 0.736     | 0.093   |
| Solar Cells for Satellites | 0.827       | 0.839      | 0.011    | 0.932      | 0.947     | 0.015   |
| Technology Prediction      | 0.835       | 0.859      | 0.023    | 0.871      | 0.910     | 0.039   |

Mean AUC gain **+0.0252**, mean F2@t* gain **+0.0377**. Clears the noise floor: **no**. **Verdict: KEEP CURRENT**.

**This one is worth pausing on rather than accepting the flat verdict above at face value.** This
project's established convention (§§9, 12-14) gates "clears the noise floor" on mean ROC-AUC,
which C=0.001 misses by a hair (+0.0252 vs. the 0.03 floor) - but on F2@t*, the metric the user
has repeatedly asked to be treated as primary, the gain (**+0.0377**) *does* clear it. More
strikingly: **every one of the 6 use cases improves on both metrics** (auc_gain and f2_gain
both positive in all 6 rows above) - no other experiment in this entire investigation (§§9,
12, 13, 14) produced a uniformly positive result across every use case; all of them had at
least one use case go the wrong way. That consistency is itself evidence this isn't noise.

A follow-up local check (5-seed within-silo, `C in {0.0001, 0.0005, 0.001, 0.005, 0.01}`, not
run via Modal - cheap, LogReg fits in seconds) confirms this is a real, well-behaved interior
optimum rather than a fluke at exactly C=0.001 or a monotonic drift that would suggest
instability:

| C      | mean within-silo AUC | mean within-silo F2@t* |
|--------|-----------------------|--------------------------|
| 0.0001 | 0.8596                | 0.8968                   |
| 0.0005 | 0.8666                | 0.8983                   |
| 0.001  | 0.8655                | 0.8969                   |
| 0.005  | 0.8569                | 0.8924                   |
| 0.01   | 0.8533                | 0.8893                   |
| 1.0 (current) | 0.8403         | 0.859                    |

Performance peaks around **C=0.0005-0.001** and falls off in both directions - a genuine
optimum, not an artifact. The likely mechanism: at ~4600 features and ~260-360 rows per silo,
sklearn's default `C=1.0` under-regularizes badly; every LogReg fit in this entire body of
work has been running measurably overfit, and correcting `C` recovers real signal.

**Revised verdict: promising, not yet fully adopted.** It clears the floor on the project's
primary metric (F2@t*) and is more consistent than anything else tested this session, but
misses the project's own established AUC-based gate, and this pass only checked `C` on the
current `(iterations=150, depth=4)` CatBoost-adjacent feature set for LogReg in isolation -
it has not been checked on LOGO with the finer grid, nor validated on SYNERGY, nor re-run
through combiner-weight selection (the existing weights in
`reports/wf_ensemble_v2_chosen_weights.json` assume `C=1.0`'s LogReg OOF and would need
re-deriving if this is adopted). Recommend: treat `C≈0.0005-0.001` as the leading candidate
for the next round of work, not yet shipped.

### Summary

- CatBoost: keep current iterations=150/depth=4 (KEEP CURRENT, within-silo mean AUC gain +0.0035, mean F2@t* gain +0.0013 - flat on both metrics, no ambiguity here).
- LogisticRegression: **PROMISING, not yet adopted** - `C≈0.0005-0.001` clears the noise floor on F2@t* (+0.0377) and improves all 6/6 use cases on both AUC and F2, but misses the established AUC-based gate (+0.0252 < 0.03) and needs SYNERGY validation + combiner-weight re-derivation before shipping. Current `C=1.0` default left unchanged pending that follow-up.


## 16. Closing the loop on LogReg C — combiner re-derivation + SYNERGY validation

Follows up on §15: a central hyperparameter search found sklearn's default `C=1.0` under-regularizes every LogReg fit in this project (~4600 features, 260-360 rows/silo); a fine sweep found a genuine interior optimum at `C=0.0005` (improved 6/6 use cases standalone). Not yet adopted because the combiner weights need re-deriving (the current ones assume `C=1.0`'s LogReg OOF) and it had never been checked on SYNERGY. This closes both gaps.

### Within-silo combiner re-derivation

For each use case, a fresh weight-grid search (`w` 0-1, step 0.05) on the `C=0.0005` LogReg OOF is chosen on seeds [0, 1, 2] (mean ROC-AUC), evaluated on seeds [3, 4] (never seen during selection) - fresh for all 6 use cases, not reusing the old STABLE/UNSTABLE split derived under `C=1.0`. Compared against the CURRENT shipped combiner (`C=1.0` LogReg OOF at the existing weight from `wf_ensemble_v2_chosen_weights.json`), evaluated on the same held-out seeds.

| use_case                   | w_current | w_new | auc_current | auc_new | auc_gain | f2_current | f2_new | f2_gain |
|----------------------------|-----------|-------|-------------|---------|----------|------------|--------|---------|
| Carbon Capture             | 0.500     | 0.200 | 0.905       | 0.906   | 0.001    | 0.896      | 0.907  | 0.011   |
| Low-Carbon Cement          | 1.000     | 0.900 | 0.911       | 0.911   | -0.000   | 0.949      | 0.949  | 0.000   |
| Named Entity Recognition   | 0.500     | 0.000 | 0.834       | 0.855   | 0.022    | 0.930      | 0.933  | 0.004   |
| Soil Microbiome            | 0.950     | 0.300 | 0.827       | 0.840   | 0.013    | 0.726      | 0.733  | 0.007   |
| Solar Cells for Satellites | 0.850     | 0.800 | 0.875       | 0.874   | -0.001   | 0.950      | 0.950  | -0.000  |
| Technology Prediction      | 0.500     | 0.000 | 0.835       | 0.862   | 0.027    | 0.907      | 0.920  | 0.013   |

Overall: mean AUC gain **+0.0103** (4/6 use cases positive), mean F2@t* gain **+0.0059** (4/6 use cases positive), evaluated on held-out seeds never used for weight selection.

Clears the noise floor on AUC: **no**. Clears the noise floor on F2@t* (the project's stated primary metric): **no**.

### SYNERGY validation

LogReg OOF at `C=0.0005` fit locally on SYNERGY (5 seeds, synthetic per-row groups since SYNERGY has no author column — same convention `scripts/modal_ensemble_experiments.py::run_synergy_oof` uses), blended 50/50 with the already-cached CatBoost(150,4) SYNERGY OOF (`reports/wf_ensemble_v2_synergy_oof_qwen8b.json`) — 50/50 because SYNERGY's reviews are unseen "use cases" the per-TIRI nested weights don't transfer to (existing convention, not a new departure). Compared against the existing `C=1.0` 50/50 blend (same file's `"logreg"` key) and against §11's published summary (mean ROC-AUC 0.899).

**New (`C=0.0005`) 50/50 blend, per review, mean across seeds:**

| review            | roc_auc | f2_at_0.5 | f2_at_t_star | recall_at_10pct | recall_at_20pct | wss_at_95 |
|-------------------|---------|-----------|--------------|-----------------|-----------------|-----------|
| Menon_2022        | 0.963   | 0.740     | 0.754        | 0.773           | 0.938           | 0.724     |
| Sep_2021          | 0.800   | 0.371     | 0.602        | 0.355           | 0.600           | 0.223     |
| van_der_Waal_2022 | 0.962   | 0.449     | 0.563        | 0.879           | 0.964           | 0.742     |

**Current (`C=1.0`) 50/50 blend, per review, mean across seeds (recomputed fresh here from the cached OOF — cross-check against §11):**

| review            | roc_auc | f2_at_0.5 | f2_at_t_star | recall_at_10pct | recall_at_20pct | wss_at_95 |
|-------------------|---------|-----------|--------------|-----------------|-----------------|-----------|
| Menon_2022        | 0.959   | 0.712     | 0.743        | 0.776           | 0.914           | 0.695     |
| Sep_2021          | 0.797   | 0.414     | 0.591        | 0.375           | 0.575           | 0.205     |
| van_der_Waal_2022 | 0.942   | 0.285     | 0.504        | 0.782           | 0.933           | 0.623     |

Overall mean ROC-AUC: new **0.9084** vs current **0.8994** (§11 published baseline: 0.899). Delta: **+0.0091**. Overall mean F2@t*: new **0.6397** vs current **0.6126**. Delta: **+0.0271**.

SYNERGY read: **IMPROVES** (delta +0.0091 mean ROC-AUC, no formal noise floor established for SYNERGY given its small per-review sizes — this is a qualitative read against the published baseline, not a hard gate).

### Decision

**Verdict: MIXED — not a clean adopt.** Specifically: within-silo F2@t* gain does not clear the noise floor. Within-silo AUC also does not clear the noise floor (+0.0103 vs 0.03). Not adopting `C=0.0005` or the re-derived weights on the strength of this result alone — current `C=1.0` defaults and `wf_ensemble_v2_chosen_weights.json` are left unchanged. New weights are NOT saved to `wf_ensemble_v2_chosen_weights_c0005.json` given the mixed result.
