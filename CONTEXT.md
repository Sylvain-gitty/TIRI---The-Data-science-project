# CONTEXT.md — what a new agent needs to know before touching this repo

`CLAUDE.md` says how to work here. This file says **what is true here**, and it exists
because several of the facts below are not derivable from the code, the data, or the git
history — they came out of a working session and would otherwise have to be rediscovered
(expensively, and in at least two cases wrongly).

Read this before proposing modelling work. Most bad suggestions in this project are bad
for one of the reasons listed here.

---

## 1. What TIRI is actually for

TIRI is one part of a larger product, not a standalone bootcamp exercise. The product goal:

- A **high-recall (F2) ensemble** that can be fine-tuned for a new use case, run over
  **100k+ papers**, at **low cost**, **100% deterministically**, on **sovereign
  infrastructure**.
- **Precision optimisation comes later.** Users incrementally label and prune toward a
  higher F1. Do not trade recall for precision in this phase.
- **Every new use case arrives with labelled data and a written brief.** How much data,
  and what brief format, are being settled by two separate experiments. Zero-label cold
  start is therefore a *transient*, not the operating regime — but see §4, because at
  realistic prevalence "arrives with labels" is a stronger assumption than it sounds.

### The constraint that decides most architecture questions

**Use cases are per-customer and siloed. There is no shared pool of papers — only shared
tooling.** Each customer's model is trained on their own labels, in their own silo.

Consequences that are easy to get wrong:

- **No pooled/global model ever ships.** What crosses the silo boundary is a *configuration*
  — feature definitions, model class, default hyperparameters, filter rules — never weights
  and never data.
- **Leave-one-use-case-out (LOGO) is a defaults-selection instrument, not a production
  estimate.** It answers "which feature families should we ship to a customer we have no
  labels for". Never quote a LOGO score as "how well TIRI works".
- **Within-silo (grouped k-fold inside one use case) is the production surface.** That is
  the number that means something.
- Hyperparameter search happens **once, centrally, in the LOGO loop**, where six use cases
  of evidence exist. Each silo then just fits weights. Per-customer grid search at a few
  hundred labels would overfit (see `reports/wf_ensemble_report.md` §4).
- Confounds that encode *use-case identity* (source provenance, `has_venue`, `year`) only
  hurt when pooling. **Inside a silo they are admissible** — several warnings in the older
  reports are pooling-specific and dissolve here.

Use cases are uniformly **technology / hard science / industry** — never social science.
That is why a shared non-topical axis (applied-vs-foundational maturity) is plausible at
all, and it is the one paper-only feature family worth testing for portability.

Pools are built from queries derived from the use case and the labelled set, plus citation
harvesting from the labelled set, then narrowed by **deterministic filtering** (not more
queries). Drift, retraining cadence and multi-labeller questions are **deliberately out of
scope right now**: the current target is a versioned model, high recall, a pinned use case,
one labeller.

---

## 2. The central technical finding

**Relevance is a property of the (brief, paper) pair, not of the paper.**

Every feature in this repo before mid-2026 described the paper alone. Measured
consequences, all reproducible:

- A LogisticRegression predicts **`use_case_key` from the paper embedding at 96.2%
  accuracy** (majority class 19%). The embedding is not noisy — it is a use-case
  fingerprint.
- So a model trained across use cases learns *which use case this is*, which is
  definitionally non-transferable. LOGO collapses to ~0.54 ROC-AUC for that reason.
- **This is not "overfitting from too many features."** Ten brief-relative scalars beat a
  384-dimension embedding on LOGO (0.642 vs 0.537). It was never the feature count; it was
  the feature *kind*.
- Strongest form: at zero labels, **unsupervised cosine-to-brief beats every supervised
  cross-domain model** (LOGO WSS@95 0.082 vs 0.042 best supervised; ROC-AUC 0.693 vs 0.610).
  A model trained on other customers is worse than no model at all.
- The corroborating clue was already in the repo: of eleven metadata features tested, the
  only one that worked was term overlap — the only one that read the brief.

**Operating rule this implies:** cosine-to-brief below ~25 in-silo labels, supervised
in-silo model above. Never a model trained on other customers' data.

Things that do **not** fix the transfer problem (both tested, both negative): per-use-case
mean-centring the embedding (0.532 → 0.529), and adding metadata.

---

## 3. Data facts that will bite you

| Fact | Why it matters |
|---|---|
| The live label is **`triage_label`**, not `review_label` (which is empty but for one row). Values: `positive` 1067, `negative` 785, `pass` 543, null 478. | 1,852 usable labelled rows. |
| **`pass` is 84.7% missing-abstract** (vs 5–7% for decided rows). Only 83 `pass` rows have an abstract. | It is a data-completeness artefact, not analyst hesitation. Dropping it is correct and keeps the model binary. Do not build an ordinal target on it. |
| `year` ranges differ wildly by use case (`ner` 2023–26, `tech_forecasting` 2025–26, `solar_leo` 1974–2026). Some rows are dated 2027. | Year is a use-case fingerprint when pooling; fine within a silo. |
| **`from_*` columns are attribution, not capability** — retrieval de-duplicates, so a paper is credited to whichever arm recorded it first. | Any analysis of "which retrieval arm could find this" using these columns is measuring the storage convention. This mistake has already been made once. |
| `terms_exclude` is **empty for 3 of 6** use cases; `performance_criteria`, `decision_*`, `trl_*` are sparse; `constraints_scale` is empty everywhere. | Features over these must emit NaN + an indicator. **NULL is not 0.** Feed the brief-format experiment: `constraints_scale` never got filled. |
| Pools run **26–77% positive**. Production will be low single digits. | Every F2 number and every calibrated threshold in the older reports was measured at ~20x production prevalence. SYNERGY (1.7–14.8%) is the only prevalence-realistic surface available. |
| `scripts/embedding_utils.py:get_use_case_text` claims the export lacks the brief and falls back to the use-case **name**. That was true once; the brief columns exist now. | The `scripts/*.py` path scores against a 2–4 word name. The notebooks build a proper brief. Do not assume they agree. |

---

## 4. Open risks nobody has closed

- **At production prevalence, a random 25-label bootstrap contains almost no positives.**
  Measured on SYNERGY's 1.7% review: **57% of random 25-label draws contain no positive at
  all** and cannot train anything (28% at n=50, 22% at n=100). "Use cases arrive with
  labels" is safe only if those labels were *actively selected* — e.g. by labelling the top
  of a cosine-to-brief ranking — not randomly sampled. Confirm how the labelling project
  sources them.
- **The query-conditioned advantage is contingent on brief format, not automatic.** The
  shuffled-brief control passes decisively on TIRI (5/6 use cases, +0.155) and **fails on
  SYNERGY** (1/3, −0.012). SYNERGY briefs are a published review's title and abstract —
  a description of what a review did, not a statement of what to include — and carry no
  curated term lists. Curated inclusion terminology looks load-bearing. Feed this to the
  brief-format experiment; re-validate the block against any new brief format.
- **Model ranking is not stable across prevalence regimes.** Qwen3-4B is mid-pack in-repo
  and *last* on SYNERGY; Qwen3-8B wins at realistic prevalence. Recall metrics only
  discriminate where there is room to skip — our 26–77% pools are a poor surface for
  judging a recall-oriented system.
- **Selection-on-holdout.** Many decisions (11 embedding models, combination methods,
  calibration methods, PCA on/off) were made against LOGO scores, so LOGO is no longer
  unbiased. **23 unused SYNERGY reviews** are the only clean surface left — ring-fence them.
- **Label recall is unmeasured.** With one labeller there is no inter-annotator agreement,
  and a labeller's false negative is indistinguishable from a true negative — it silently
  inflates measured recall. Same problem applies to the deterministic filter's discards.
  One fix covers both: sample the negatives (and the filter's rejects) and re-label blind.
- **No causal account of `solar_leo`'s labels.** They track publication year because the
  pool was seeded from citation-ranked canon. Treat it as a corpus defect: never cite it as
  evidence the system finds relevant work.

---

## 5. Statistical discipline this repo requires

Six use cases, 260–360 labelled rows each, per-use-case scores spanning 0.26–0.85. That is
a small, heterogeneous sample and it has already produced at least one decision made on
noise.

- **Measured noise floors:** seed-to-seed sd is ~0.010 on within-silo ROC-AUC and
  **0.015–0.027 on WSS@95**. The entire spread between the three candidate embedding models
  is 0.027 — one sd. Treat any gap below ~0.03 as *not established*.
- Report **per-use-case win counts** alongside means. A mean over six heterogeneous use
  cases can be won by being good at the easy ones.
- Repeat across seeds and report the spread next to every number.
- Never make an architecture decision from a single held-out use case. Several numbers in
  `reports/wf_embedding_bakeoff.md` Rounds 1–2 rest on `tech_forecasting` alone.

---

## 6. The negative-results register

The most valuable asset here. Measured and rejected, so nobody re-runs them:

| Rejected | Evidence |
|---|---|
| The 11-feature metadata punch list (citation velocity, author count, has_venue, is_english, venue cleaning…) | `reports/wf_featureengineering_review.md` — all together worth +0.002 AUC, and they *hurt* out-of-domain |
| TRL keyword estimate, OpenAlex venue quality, author ORCID | earlier notebooks in `notebooks/experiments/` |
| `relevance_score` as a feature or baseline | `reports/wf_ensemble_report.md` §0 — unversioned, moving, not comparable across use cases |
| Prediction-level stacking / learned blending | tied with plain averaging, twice (`wf_ensemble_report.md`, `wf_embedding_bakeoff.md` §5) |
| spaCy over plain regex for term matching | no gain, more cost |
| SPECTER2 | query mode mismatched to paragraph-length briefs |
| Per-use-case mean-centring of embeddings | 0.532 → 0.529 LOGO, no effect |
| **PCA-64 within a silo** | −0.008 to −0.014 WSS on all three models; §6.2's compress-then-concat win was a *transfer* phenomenon and does not carry to production folds |
| Term overlap as an abstract-length proxy (a suspicion, now closed) | length features alone reach 0.550; removing them costs 0.004 |

---

## 7. Where the current work lives

| What | Where |
|---|---|
| Query-conditioned lexical features (Tier 1b) + falsification control | `scripts/lexical_features.py`, `scripts/run_tier1b_control.py` |
| Jasper / Qwen3-4B / Qwen3-8B on WSS@95, both fold surfaces | `scripts/run_embedding_recall_comparison.py` |
| SYNERGY validation at realistic prevalence | `scripts/run_synergy_recall_validation.py` |
| Shared fold/metric helpers (use these, do not re-implement) | `scripts/fold_pipeline_utils.py`, `scripts/embedding_utils.py` |
| Full narrative of the above | `reports/wf_query_conditioned_findings.md` |
| Ensemble v1 — per-silo CatBoost + LogisticRegression, feature/embedding ablation, and what was cut from the original proposal | `reports/wf_ensemble_v1_candidate.md`, `reports/wf_ensemble_v1_results.md`, `notebooks/main/09_ensemble_per_silo.ipynb` |
| CatBoost fitting on this machine — route through Modal (`scripts/modal_ensemble_candidate.py`), do not fit locally | `scripts/ensemble_eval_utils.py`'s consumers; see that file's module docstring for the confirmed Apple Silicon thread-oversubscription pathology |
| Ensemble v2 — hyperparameter tuning, nested combiner-weight selection, the Qwen3-8B SYNERGY swap, a 3-lever diversity sweep (SVM/lexical-only/k-NN as a third branch, all rejected, each for a documented reason), and a LOGO-based central hyperparameter search (LogReg `C=1.0` found under-regularizing; not adopted for the 6 shipped use cases but recommended as the starting default for new ones) | `reports/wf_ensemble_v2_experiments.md` (the full running log, §1-16); Modal functions consolidated in `scripts/modal_ensemble_experiments.py` — **do not split Modal functions across files**, see that file's docstring |
| Final, synthesized architecture recommendation — one decision doc pulling together v1 + v2, confidence-graded, with explicit rejects and caveats | `reports/wf_ensemble_final_recommendations.md` |

**The shuffled-brief control is the pattern to copy.** Any feature claiming to read the
brief must be rebuildable against deliberately wrong briefs. If it still scores well, it is
measuring something generic and should be thrown away. `build_lexical_features(df,
brief_map=...)` exists for exactly this reason — keep that seam in anything you add.
