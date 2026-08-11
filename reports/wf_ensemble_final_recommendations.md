# Ensemble — final recommendations

**What this is:** the single decision doc pulling together everything from the
ensemble-modelling plan, Phase 1-3 (`reports/wf_ensemble_v1_candidate.md`, `wf_ensemble_v1_results.md`,
`notebooks/main/09_ensemble_per_silo.ipynb`), and the v2 exploratory round
(`reports/wf_ensemble_v2_experiments.md`). Each line below is a decision, not a discussion —
read the linked report for the full argument.

**Confidence key:** 🟢 externally validated or clears the multi-seed noise floor · 🟡 real,
small, honestly measured but doesn't clear the noise floor · ⚪ engineering/process choice,
not a performance claim.

---

## The architecture

| # | Decision | Rationale | Evidence |
|---|---|---|---|
| 1 | 🟢 **Per-silo, never pooled.** One model per use case, fit only on that use case's own labels. No shared weights or data across customers, ever. | A pooled LogisticRegression reads `use_case_key` off the raw embedding at 96.2% accuracy — a pooled model learns "which customer is this," not relevance. | `CONTEXT.md` §1-2 |
| 2 | 🟢 **Two branches: CatBoost (Ordered boosting) + LogisticRegression**, same feature set, combined by weighted average. | RandomForest/HistGradientBoosting hit train AUC = 1.000 under LOGO (severe leakage); CatBoost's Ordered boosting specifically suppresses it. LightGBM shares the same leakage mechanism and was never substituted in. Stacking/learned combiners tied plain averaging twice already in this repo's history. | `scripts/fold_pipeline_utils.py` `build_tree_pipeline` docstring; `CONTEXT.md` §6 |
| 3 | 🟢 **Features: Tier-1b lexical BM25 subset (10 cols, not the full 22) + cosine-to-brief (per embedding model) + 5 raw metadata columns + the raw embedding block, uncompressed.** | BM25 alone carries 0.637 of the full 22-column block's 0.642 LOGO score — the other 12 columns aren't earning their place at this sample size. Metadata (`year`, `paper_age`, `has_abstract`, `n_authors`, `citation_count`) kept because it's free, not because it's expected to matter (`author_count` alone tested as a coin flip). | `reports/wf_query_conditioned_findings.md` §4; `reports/wf_featureengineering_review.md` §4.1 |
| 4 | 🟢 **Embedding: Jasper + Qwen3-8B concat** (not Jasper + Qwen3-4B). | Directly tested via SYNERGY swap: fixed the one review where the ensemble lagged the published baseline (ROC-AUC +0.046, WSS@95 +0.122 on Sep_2021), left the other two unchanged, and the swapped ensemble now matches or beats the best single-embedding baseline ever measured here on all three reviews (mean ROC-AUC 0.899 vs. 0.888). | `reports/wf_ensemble_v2_experiments.md` §11 |
| 5 | 🟢 **CatBoost hyperparameters: `iterations=150, depth=4`** (not the `50/4` screening setting, and not pushed further to `300`). | Multi-seed check: `50→150` is a real, per-use-case gain on 3-4/6 use cases (largest where sd is tightest — Soil Microbiome, Solar Cells, Carbon Capture). `150→300` showed no further gain (+0.0015, flat) — diminishing returns already reached. **Now also confirmed via a proper central search**: a 20-config depth×iterations grid searched via LOGO (as `CONTEXT.md` §1 specifies for this exact purpose, not the within-silo surface the checks above used) found a LOGO-preferred setting (`iterations=350, depth=3`), but validated on fresh within-silo 5-seed OOF it was flat against the current setting (AUC +0.0035, F2@t* +0.0013 — under the noise floor). Current setting kept. | `reports/wf_ensemble_v2_experiments.md` §§6-8, 15 |
| 6 | 🟡 **Combiner: 50/50 for Carbon Capture, Named Entity Recognition, and Technology Prediction; weight CatBoost ~0.85-1.0 for Soil Microbiome, Low-Carbon Cement, and Solar Cells for Satellites.** | The weight-selection pattern is stable (low seed-to-seed variance) for exactly these three use cases and unstable (bounces across the whole 0-1 range) for the other three — a real, reproducible split, not an artifact. Properly re-validated with the weight chosen on one set of seeds and scored on a held-out set never used for selection: honest gain +0.002 to +0.013 F2/AUC, never negative. | `reports/wf_ensemble_v2_experiments.md` §§3, 5, 9 |

## Deployment / operating point

| # | Decision | Rationale | Evidence |
|---|---|---|---|
| 7 | 🟢 **Use Recall@k% of the reviewed pool as the primary operating metric — not a fixed probability threshold.** | Current pools run 26-77% positive; production will be low single digits (~20x lower). A probability threshold tuned here does not transfer across that big a prevalence shift; rank order is far more stable. | `CONTEXT.md` §1/§3; `reports/wf_ensemble_report.md` §2 |
| 8 | 🟢 **Cold-start ladder for a new use case: 0-24 labels → ship the zero-label cosine-to-brief ranker only (and use its top ranks to actively source the next labels, not randomly); ≥25 labels → switch to the full per-silo ensemble.** | Supervised in-silo overtakes the zero-label ranker right around n=25 (warm-start curve). Random label sampling is unsafe at low prevalence — 57% of random 25-label draws contain zero positives at 1.7% prevalence (SYNERGY). | `reports/wf_query_conditioned_findings.md` §2/§7 |
| 9 | 🟡 **A brand-new use case's initial LogReg branch should start at `C≈0.0005`, not sklearn's default `C=1.0`** — the 6 already-shipped use cases keep their current settings unchanged. | Closing the loop on the central search's LogReg finding produced a genuine surface split, not a clean win: on within-silo (the 6 already-tuned customers, `CONTEXT.md`'s stated production surface) the re-derived combiner gain doesn't clear the noise floor on either AUC (+0.010) or F2 (+0.006) — CatBoost's own contribution already subsumes much of what `C=0.0005` fixes once optimally blended. But on SYNERGY — whose reviews are structurally the same situation as a brand-new customer (no per-silo history) — it's a real, consistent win (ROC-AUC +0.009, F2@t* +0.027, positive on all 3 reviews). Read together: not proven to help customers we already have deep history with, but the better starting point for one we don't yet. Not yet stress-tested at an actual ~25-40-label cold-start scale (SYNERGY's reviews are much larger) — a reasonable inference from the best available evidence, not a fully closed loop of its own. | `reports/wf_ensemble_v2_experiments.md` §§15-16 |

## Explicitly rejected — do not re-propose these

| Rejected | Why |
|---|---|
| Matryoshka truncation, a learned low-rank bilinear layer, Hadamard/diff PCA-sliced interactions | PCA-64 on the raw embedding measured hurting within-silo (−0.008 to −0.014 WSS@95) even though it helps LOGO/transfer — a supervised single-axis variant (add-alongside) was tested fresh in v2 and does nothing either (§8) |
| LightGBM | Shares RF/HGB's leakage mechanism; CatBoost Ordered was chosen specifically to suppress it |
| Streaming `SGDClassifier`/`partial_fit` | Real complexity solving a problem that doesn't exist at 260-360 rows/silo; full refit is sub-second |
| Flat `scale_pos_weight × 2.0` | Per-silo prevalence already ranges 26-77%; some use cases are majority-positive, where this would push further off-balance. `class_weight="balanced"` + threshold-stage recall preference is the existing, consistent convention |
| Blanket Platt calibration | Multi-seed check: mean lift ≈ 0 (noise 50x the mean), and *negative* on Soil Microbiome specifically — the single-seed "it helps a little" read didn't survive repetition |
| A fixed global F2 threshold | Measured at ~20x production prevalence; won't transfer. Recall@k is the correct primary metric (#7 above) |
| DeLong significance test | This repo's own measured noise floor (~0.010 ROC-AUC seed-to-seed sd) is a sharper, pre-calibrated bar than a bare p<0.05; `bootstrap_auc_ci` (already implemented) is reused instead |
| A third ensemble branch: SVM (RBF kernel) | Standalone it's actually the strongest single branch on 4/6 use cases — but it's *more* correlated with both existing branches (ρ 0.86-0.91) than CatBoost and LogReg are with each other (ρ 0.78, the pair already in production). A properly nested 3-way blend (weights chosen on one seed set, scored on a held-out set) gains only +0.010 mean AUC — under the 0.03 noise floor — and mean F2@t* is slightly *negative*. Diversity problem, not a capability gap: a third branch on the same ~4600-dim feature set buys a stronger echo, not a new opinion |
| A third ensemble branch: lexical/metadata-only, no embedding | Genuinely more decorrelated this time (ρ 0.52-0.67 vs. existing branches vs. their own 0.78 with each other) — confirms the *diversity* mechanism is real when the feature view actually changes. But standalone it's much weaker on 4/6 use cases (it drops the whole embedding block), so the disagreement is too weak a vote to move the blend: mean nested AUC gain +0.006, mean F2@t* gain *negative* (-0.003, driven by a -0.033 hit on Technology Prediction). Sample-size/ceiling problem, not a diversity problem, this time — see the Low-Carbon Cement exception below |
| A third ensemble branch: k-NN (cosine, k=31) | Tests the one remaining lever — same feature set, but a mechanistically different (local, non-boundary-fitting) algorithm. Correlation is lower than SVM's (ρ 0.87/0.74 vs. SVM's 0.91/0.86 — the mechanism difference is real) but its mean (0.81) is still *higher* than the existing pair's own 0.78, and standalone it's not the best branch on any single use case. Mean nested AUC gain +0.003, mean F2@t* gain *negative* (-0.005). **This closes the 3-lever sweep** (algorithm-only → SVM, feature-view-only → lexical, mechanism-only → k-NN) at reject for all three, each for a distinct, documented reason: this feature space is separable enough that neither a different algorithm, nor a different mechanism, disagrees enough to help — only a different *feature view* disagrees enough, and that view isn't strong enough standalone (except at Low-Carbon Cement) |

## Worth a further look, not yet adopted

- **Replacing the raw ~6000-dim embedding block with a single supervised discriminant
  direction scored statistically identical** (F2@t* 0.897 vs. 0.892) at a fraction of the
  compute/serving cost. Not a performance change, but a real simplification opportunity —
  worth validating on SYNERGY before adopting, since it hasn't been checked there yet.
  (`reports/wf_ensemble_v2_experiments.md` §8)
- **Re-running the lex/cos-brief/metadata ablation under the now-tuned CatBoost** — a
  stronger base learner can sometimes absorb what weaker engineered features were
  compensating for; never explicitly re-checked after the hyperparameter tuning landed.
- **A third embedding branch (Qwen3-8B alone, or a 3-way blend)** for additional diversity —
  flagged early, never tested; lower priority now that the Qwen3-8B swap already closed the
  main gap.
- **A real, central hyperparameter search** (not the two-point 50→150→300 sweep used here)
  for both CatBoost and LogisticRegression, run once via the existing LOGO harness, per
  `CONTEXT.md` §1's "search happens once, centrally" rule.
- **Low-Carbon Cement specifically: a lean (lexical+metadata-only, no embedding) CatBoost
  model alone beat *both* current production branches standalone** (ROC-AUC 0.933 vs. 0.912
  CatBoost-full / 0.876 LogReg-full), and a nested weight search put 100% of the blend weight
  on it for this one use case — the largest single gain in the whole lean-branch experiment
  (+0.023 AUC, +0.006 F2@t*, on held-out seeds). This surfaced *because* of caveat #5 below
  (this branch pair's ρ≈0.88 borders on redundant) — worth a dedicated, single-use-case look
  rather than folding into the general "third branch" verdict, which was a reject.
  (`reports/wf_ensemble_v2_experiments.md` §13)

---

## Caveats and tradeoffs to hold in mind

1. **Everything here is still measured on enriched pools (26-77% positive), not production
   prevalence (low single digits).** SYNERGY (1.7-14.8%) is the best proxy available and the
   embedding swap was validated there — but it's still higher-prevalence than a likely real
   deployment, and thresholds (not rankings) are the part most at risk of not transferring.
2. **The SYNERGY validation tested a reduced feature set** — no `cos_brief`/metadata (not
   available for SYNERGY's briefs/reviews), just lexical BM25 + raw embedding. The full
   recommended feature set (with cos_brief + metadata) has only been validated in-repo, not
   externally.
3. **The per-use-case combiner weights (#6) are specific to these six named use cases.**
   They do not transfer to a new customer — a new silo starts at 50/50 (or better, the
   cold-start ladder in #8) until it has enough labels to justify its own weight check.
4. **CatBoost hyperparameters (150/4) are a reasonable middle ground found under real
   compute constraints, not an exhaustive search.** Depth in particular costs far more than
   a naive iterations×depth estimate suggested — confirmed the hard way (a `depth=6` attempt
   blew a 1-hour Modal timeout). Treat 150/4 as "solidly better than the screening default,"
   not "optimal."
5. **Branch correlation on Low-Carbon Cement (ρ≈0.88) borders on the redundant end of the
   healthy range** — worth re-checking as more labels accumulate; not yet a reason to drop
   either branch there.
6. **Engineering note, not a modelling tradeoff:** CatBoost is pathologically slow on this
   development machine's Apple Silicon (a confirmed thread-oversubscription bug, not a real
   workload cost) — every CatBoost fit in this entire body of work ran on Modal
   (`scripts/modal_ensemble_experiments.py`), never locally. Anyone continuing this work on
   the same machine needs to do the same.
7. **The weighted-combiner gain (#6) and the calibration rejection (explicitly-rejected
   table) both came from single-round, then-repeated checks — not the same 5-seed-on-every-
   claim discipline Phase 1 used for the architecture itself.** They're the most-validated
   version available, but a further repeat (different seed ranges, more seeds) would still
   sharpen confidence further before this is treated as fully settled.
