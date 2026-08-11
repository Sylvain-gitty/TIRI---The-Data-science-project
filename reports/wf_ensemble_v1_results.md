# Ensemble v1 — what shipped, what was cut, and why

**Audience:** anyone picking up TIRI's modelling work next. Companion to the approved
ensemble-modelling plan (a per-silo CatBoost + LogisticRegression ensemble on
`data/processed/papers_fe.parquet`) and its two artefacts: `reports/wf_ensemble_v1_candidate.md`
(Phase 1, the feature/embedding/branch ablation) and
`notebooks/pipelines/wf_ensemble_fold_pipeline.ipynb` (Phase 2, the reviewable walkthrough).
This report is Phase 3: the decision trail, not a repeat of either artefact's numbers.

---

## 1. TL;DR

**What shipped:** per use case (never pooled — `CONTEXT.md` §1), a CatBoost (Ordered
boosting) + LogisticRegression ensemble on Tier-1b lexical (BM25 subset) + cosine-to-brief +
5 raw metadata columns + one raw embedding block, combined by simple 50/50 averaging.
Clears the published Tier1b+embedding LogisticRegression baseline
(`reports/wf_tier1b_lexical_control.md` §2) by +0.09 to +0.18 ROC-AUC across use cases —
decisively past the ~0.03 noise floor (CONTEXT.md §5).

**What was proposed but cut, and why:** Matryoshka truncation, a learned low-rank bilinear
interaction layer, Hadamard/diff PCA-sliced interactions, LightGBM, a streaming
`partial_fit` branch, a flat `scale_pos_weight × 2.0` heuristic, and a fixed F2 probability
threshold. Every cut traces to a specific number already measured in this repo — §3 below.

**The one open tie:** Qwen3-8B vs. Jasper+Qwen3-4B concat as the embedding block is a
near-coin-flip (0.838 vs. 0.846 mean within-silo ROC-AUC, CatBoost) — consistent with
`reports/wf_query_conditioned_findings.md` §5's finding that the three candidate embeddings
are statistically tied within a silo. The notebook picks the higher mean programmatically;
don't read more into the gap than that.

**Engineering surprise worth flagging for whoever touches this next:** CatBoost's default
`thread_count=-1` is pathologically slow on this development machine's Apple Silicon (a
single 5-feature, 264-row fit never finished in 10+ minutes; `thread_count=1` took ~7s).
Every CatBoost fit in this work runs on Modal instead — see §4.

---

## 2. What shipped, and the numbers behind it

| Component | Choice | Evidence |
|---|---|---|
| Scope | Per use case, independent fits, never pooled | `CONTEXT.md` §1 — a pooled LogisticRegression reads `use_case_key` off the raw embedding at 96.2% accuracy |
| Branch A | CatBoost, `boosting_type="Ordered"` | Already the repo's chosen tree model (`fold_pipeline_utils.build_tree_pipeline`) — RF/HGB hit train AUC 1.000 under LOGO |
| Branch B | LogisticRegression | The already-tested baseline learner throughout `reports/wf_tier1b_lexical_control.md` |
| Features | `lex_bm25_*`/`lex_rank_bm25_*` (10 cols) + `cos_brief_*`/`rank_cos_brief_*` (2-6 cols) + 5 metadata cols + 1 raw embedding block | BM25 alone is 0.637 of the full 22-col Tier1b block's 0.642 (`wf_query_conditioned_findings.md` §4) |
| Embedding | Jasper+Qwen3-4B concat (picked programmatically; Qwen3-8B is a near-tie) | Phase 1 candidate report §2-3 |
| Combiner | Simple 50/50 average of OOF probabilities | `CONTEXT.md` §6 — prediction-level stacking tied with averaging, twice already |

Phase 1 gate (within-silo ROC-AUC delta vs. the published Tier1b+embedding LogisticRegression
baseline, `reports/wf_ensemble_v1_candidate.md` §3):

| Feature set | CatBoost | LogisticRegression |
|---|---|---|
| BM25 subset only | 1/6 clear, mean −0.034 | 0/6 clear, mean −0.030 |
| + cos-brief + metadata (no embedding) | 3/6 clear, mean +0.036 | 4/6 clear, mean +0.038 |
| + Qwen3-8B | **6/6 clear, mean +0.097** | **6/6 clear, mean +0.098** |
| + Jasper+Qwen3-4B concat | **6/6 clear, mean +0.105** | 5/6 clear, mean +0.099 |

Phase 2 diagnostics on the winning combination (`reports/wf_ensemble_fold_pipeline_7_summary.csv`):

| use_case | n | prevalence | ensemble ROC-AUC | Recall@20% | F2 @ t* | branch corr. (r) |
|---|---|---|---|---|---|---|
| Carbon Capture | 295 | 0.495 | 0.887 | 0.390 | 0.888 | 0.833 |
| Low-Carbon Cement | 262 | 0.649 | 0.906 | 0.294 | 0.948 | 0.880 |
| Named Entity Recognition | 312 | 0.715 | 0.836 | 0.274 | 0.932 | 0.698 |
| Soil Microbiome | 357 | 0.263 | 0.816 | 0.511 | 0.716 | 0.681 |
| Solar Cells for Satellites | 358 | 0.768 | 0.846 | 0.258 | 0.948 | 0.608 |
| Technology Prediction | 264 | 0.595 | 0.841 | 0.318 | 0.896 | 0.750 |

**F2 (the product's stated optimisation target, `CONTEXT.md` §1) — the direct answer to
"how does this perform for F2":** 0.72–0.95 across the six use cases at each one's own
optimal threshold (t* = 0.10–0.17, all well below the naive 0.5 cutoff — recall-weighted
scoring pushes the operating point toward "flag more, screen more" as expected). At the
naive 0.5 threshold F2 is measurably worse everywhere (0.54–0.89) — Soil Microbiome is the
biggest gap (0.54 → 0.72), consistent with it being the lowest-prevalence (26%) and one of
the two hardest use cases per `reports/wf_featureengineering_review.md` §5's diagnosis
(relevant/irrelevant papers share the same topic there, so nothing that reads topic —
including this ensemble — separates them cleanly). **The caveat that matters more than any
of these numbers:** every one of them is measured at this pool's current 26–77% prevalence.
Production prevalence is low single digits (`CONTEXT.md` §1/§3) — roughly 20x lower — and F2
at a fixed threshold does not transfer across a prevalence change like that. Don't ship any
of these t* values; re-derive on a prevalence-realistic sample (SYNERGY, or real deployment
labels) first, or lean on the Recall@k operating point instead, which is far less
prevalence-sensitive to read.

Branch correlation lands in the target 0.40–0.85 band on 5/6 use cases (Low-Carbon Cement, at
0.880, is borderline-redundant — worth watching, not yet worth dropping a branch over one
use case). The 50/50 ensemble matches or beats both individual branches on 3/6 use cases —
a free, simple win on those, a wash (not a loss) on the rest. Calibration is close to the
diagonal on 4/6 use cases; Named Entity Recognition (+0.031) and Solar Cells for Satellites
(+0.036) show a real, quantified — not just visually-bowed — systematic underconfidence,
worth a Platt-scaling pass before either use case ships a probability threshold.

---

## 3. What was cut, and the evidence for each cut

The original proposal this plan reviewed included several components that conflict with
findings already measured in this repo. None were cut on style preference — each traces to
a specific number:

| Cut | Why | Evidence |
|---|---|---|
| Matryoshka truncation, a low-rank bilinear interaction layer, Hadamard/diff PCA-sliced interactions | PCA-64 is the closest tested analogue and it hurts within a silo — compression helps LOGO/transfer (destroys use-case-identifying directions) and hurts production (throws away information with nothing to protect against) | `reports/wf_query_conditioned_findings.md` §5: "Do not carry PCA into production on the strength of a LOGO result" |
| LightGBM as the tree branch | RandomForest/HistGradientBoosting already hit train AUC = 1.000 under LOGO from the same generic-GBDT target-leakage mechanism LightGBM shares; CatBoost's Ordered boosting was chosen specifically to suppress it | `scripts/fold_pipeline_utils.py`'s `build_tree_pipeline` docstring |
| A single pooled model across all 6 use cases | A pooled model reads `use_case_key` off the embedding at 96.2% accuracy — it learns which use case a paper belongs to, not relevance | `CONTEXT.md` §1-2 |
| A streaming `SGDClassifier`/`partial_fit` branch | At 260-360 rows per silo, a full refit is sub-second; `partial_fit` trades that non-problem for real complexity (batch-consistent scaling, no native missing-value handling) | Judgement call, not a measured rejection — revisit only if a silo's row count reaches the tens of thousands |
| Flat `scale_pos_weight = neg/pos × 2.0` | Per-silo prevalence already ranges 26-77% positive; `solar_leo` is majority-positive, so doubling its positive weight would push it further off-balance, not correct an imbalance | `CONTEXT.md` §3 (per-use-case prevalence table) |
| A fixed F2 probability threshold (t*≈0.18-0.28) | Current pools run 26-77% positive; production is low single digits — any threshold fit on today's data is measured at ~20x production prevalence and won't transfer | `CONTEXT.md` §1/§3; `reports/wf_ensemble_report.md` §2 recommends Recall@k instead |
| A DeLong significance test | This repo's own measured noise floor (seed-to-seed sd ~0.010 ROC-AUC, 0.015-0.027 WSS@95) is a sharper, already-calibrated bar than a bare p<0.05 | `CONTEXT.md` §5; `fold_pipeline_utils.bootstrap_auc_ci` reused instead |

---

## 4. Engineering note: CatBoost on this machine, and why Modal is now part of the pipeline

Every CatBoost fit in Phase 1 and Phase 2 runs on Modal (`scripts/modal_ensemble_candidate.py`,
app `tiri-ensemble-ablation`), not locally. This wasn't a design preference — it was forced:

- A single CatBoost fit (5 features, 264 rows, default `thread_count=-1`) never completed
  in 10+ minutes on this development machine (Apple Silicon, macOS 15.7.5). Pinned to
  `thread_count=1`, the same fit took ~7s — a severe thread-oversubscription pathology, not
  a real workload cost.
- Even routed through Modal (ordinary x86_64 CPUs), the widest feature sets (~4600 raw
  embedding columns) needed real tuning to fit in a reasonable window: `iterations` cut from
  CatBoost's default 1000 to 50, `depth` from 6 to 4, and the per-cell timeout raised twice
  (1200s → 1800s → 2400s) before the full 150-fit (5 seeds × 5 folds × 6 use cases) sweep
  completed cleanly. This is a genuine per-fit cost at this feature width, not a memory
  problem — an 8GB/8-CPU container hit the identical wall at 100 iterations.
- `iterations=50, depth=4` is a **screening setting**, chosen to make the 8-cell ablation
  grid tractable, not a tuned final hyperparameter. Whoever moves this ensemble from
  "validated" to "shipped per customer" should revisit both knobs for the one feature set
  this settled on, with proper hyperparameter search — centrally, once, per `CONTEXT.md` §1's
  "hyperparameter search happens once, centrally" rule — not per silo.

## 5. What's still open

1. ~~Embedding choice is a coin-flip, not a decision.~~ **Updated by `reports/wf_ensemble_v2_experiments.md`
   §10:** the SYNERGY check this predicted as necessary has now been run. The
   Jasper+Qwen3-4B ensemble underperforms the published Qwen3-8B-alone baseline specifically
   on the smallest, highest-prevalence SYNERGY review (Sep_2021: 0.751 vs. 0.798 ROC-AUC),
   consistent with `reports/wf_query_conditioned_findings.md` §5's finding that Qwen3-4B is
   *last* of the three candidates at realistic prevalence. **Swapping Qwen3-4B for Qwen3-8B
   in the embedding block is now the clearest concrete next test**, not just a flagged
   uncertainty.
2. **Recall@k/WSS@95 in this report is measured at 26-77% pool prevalence, ~20x production.**
   Re-measure the operating point on SYNERGY or an actively-prevalence-matched sample before
   quoting a Recall@k number as a production expectation.
3. **Named Entity Recognition and Solar Cells for Satellites are systematically underconfident**
   (calibration bias +0.031 / +0.036) — a Platt-scaling pass on those two specifically before
   either ships a threshold-based decision.
4. **Low-Carbon Cement's branch correlation (0.880) borders on redundant.** Not yet worth
   dropping a branch over one use case, but worth re-checking once more labels accumulate.
5. **Central hyperparameter search for the real CatBoost/LogisticRegression settings** (not
   the screening `iterations=50/depth=4`) hasn't happened yet — the natural next LOGO-loop
   task, and a candidate for Modal's parallel `.map()` if the trial count makes it slow
   locally.

---

## 6. Where the evidence lives

| What | Where |
|---|---|
| Feature/embedding/branch ablation, full numbers | `reports/wf_ensemble_v1_candidate.md` |
| Reviewable, stage-by-stage walkthrough of the winning combination | `notebooks/pipelines/wf_ensemble_fold_pipeline.ipynb` |
| Shared LOGO/within-silo eval harness (used by both the ablation and `run_tier1b_control.py`) | `scripts/ensemble_eval_utils.py` |
| Modal execution (why, and the deploy/run commands) | `scripts/modal_ensemble_candidate.py` module docstring |
| The approved plan this work executes, with full evidence-cited corrections to the original proposal | `/Users/warrenfauvel/.claude/plans/rustling-gliding-creek.md` |
| Project-wide constraints and negative-results register this all builds on | `CONTEXT.md` |
