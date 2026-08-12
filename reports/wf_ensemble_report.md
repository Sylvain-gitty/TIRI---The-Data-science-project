# Ensemble strategy recommendations

> **Status: historical** — written before Ensemble v1 existed. Superseded on architecture by [`wf_ensemble_final_recommendations.md`](wf_ensemble_final_recommendations.md). Still the source of record for two entries in `CONTEXT.md`'s negative-results register: §0 (why `relevance_score` is unusable as a feature or a baseline) and §4 (why per-customer hyperparameter search overfits at a few hundred labels).

Recommendations for Week-3 modelling, assuming the feature engineering in
`reports/wf_eda_fe_report.md` gets built. Written after new facts about `relevance_score`
changed one of that report's recommendations — see §0 before reading anything else here
or in `notebooks/experiments/wf_pre_pipeline_checks.ipynb` §1.

---

## 0. Correction to prior guidance: `relevance_score`

**What it actually is** (new information, not previously known): cosine similarity
between the use case's own query text and a paper's title+abstract, computed within
that use case's own pool. It is:

- **Ordinal, not a probability** — its rank order within a pool is meaningful, its
  absolute value isn't calibrated to anything.
- **Not comparable across use cases** — two different embedding models are live across
  use cases, with a 0.265 mean gap between them. A `relevance_score` of 0.6 in one use
  case and 0.6 in another are not the same statement.
- **A moving, unversioned target.** The refinement loop that updates it is intentional
  online learning on the `academic_research_agent` side, not a defect — but it means the
  score for the *same paper in the same use case* can change over time as the spec or
  embeddings get refined, with no version marker distinguishing an old score from a new
  one in the export we currently pull.

**What this corrects:** `reports/wf_eda_fe_report.md` recommended adding `relevance_score`
as an ensemble input, and `notebooks/experiments/wf_pre_pipeline_checks.ipynb` §1 reported it
"generalizing better than the embedding ensemble" (holdout AUC 0.649 vs. 0.530–0.586)
and treated that as evidence it was a useful cross-domain signal. Both are wrong, for
related reasons:

1. **The comparison was apples-to-oranges, not a generalization result.** The embedding
   ensemble was trained on 5 use cases and evaluated on a genuinely unseen 6th — a real
   transfer test. `relevance_score` was never "trained" on anything; it's freshly
   computed per use case from that use case's own text against its own pool. Scoring
   well on `tech_forecasting` shows the cosine-similarity heuristic is locally
   reasonable *for that use case*, not that it transfers knowledge across domains. The
   numbers in `wf_pre_pipeline_checks.ipynb` §1 are still arithmetically correct; the
   "generalizes better" interpretation drawn from them is retracted.
2. **Even setting that aside, it now fails the versioning bar independently.** A model
   feature or evaluation baseline needs to mean the same thing every time it's computed.
   `relevance_score` doesn't, given the live refinement loop and the two-embedding-model
   split.

**Rules going forward:**

- **Don't use `relevance_score` as a model feature.** It will drift under the model as
  the live scoring changes, silently invalidating anything trained on it.
- **Don't use it as an independent baseline when evaluating retrieval.** Comparing a
  candidate model against a moving target produces a result that can't be reproduced
  next month, and can't be compared across use cases anyway.
- **If a relevance-style feature is genuinely needed, source it from a frozen spec
  version** — a specific, pinned `use_case_version` snapshot, never the live score.

---

## 1. Evaluation strategy: what to evaluate against, and how to avoid the version trap

**Clean, fixed-criteria evaluation sets** — use these for anything you want a stable,
reproducible number from:

- **uc1** — no spec refinement history, so no contamination from the drift described in
  §0. *Action item: confirm this use case's data is available to TIRI and pin its
  export.*
- **SYNERGY** — an external, independent benchmark: 26 published systematic reviews,
  ~169k works, ~1.7% included, with inclusion/exclusion decided by each review's fixed,
  pre-registered protocol — nothing in it is downstream of `academic_research_agent`'s
  live process at all. Genuinely useful as a check that the modelling approach
  (embeddings + engineered features + ensembling) generalizes to ground truth this
  project had no hand in shaping, not just to more `academic_research_agent` exports.
  Free and public: [github.com/asreview/synergy-dataset](https://github.com/asreview/synergy-dataset).
  *Action item: pull it in as a wholly separate validation set, evaluated with the same
  fold discipline as everything else, kept clearly out of any training pool.*

**For the 6 existing use cases (and any future one that isn't uc1/SYNERGY): stratify by
`use_case_version` and `embed_model`.** Both are confounds hiding inside what currently
looks like one flat table — a use case's spec can get refined mid-collection, and which
embedding model produced a given row's vector can differ. Any performance comparison
that doesn't stratify by these risks attributing a version/model artifact to a real
modelling effect.

**A gap to close before this is fully actionable:** `use_case_version` is not currently
a column anywhere in this pipeline. Checked directly — it's absent from both the JSONL
`_meta` header (`embed_model`, `dim`, `exported_at`, `app_version`, `use_case`) and
`usecase.json`'s schema (`schema_version` there is the *file format* version, not a
content/spec version counter). `embed_model` **is** already on every row of
`papers_combined.parquet`, but it happens to be a single constant value across the
current snapshot (`paraphrase-multilingual-MiniLM-L12-v2@fastembed-0.8.0` for all 2,873
rows) — the two-live-models situation described in §0 isn't active in our current data,
but the pipeline needs to be built assuming it will be, not re-built later when it
suddenly is. *Action item: once `use_case_version` appears in a fresh export,
`notebooks/main/01_data_compile.ipynb` needs a broadcast step for it, same as
every other `_meta`/`usecase.json` field; until then, treat any cross-time comparison
on the current 6 use cases as unstratified and provisional.*

**What this means for baselines specifically:** with `relevance_score` disqualified
(§0), the reference points for "is our model any good" become: a majority-class
baseline, performance on the frozen uc1/SYNERGY sets, and — for tracking whether a new
model version is actually better — your own prior model version, explicitly pinned.
Version everything you compare against, not just the training data.

---

## 2. Metrics

Unchanged from the previous recommendation, restated briefly since it doesn't depend on
the `relevance_score` correction:

- **Primary: Recall@k as a % of the pool reviewed** (not a fixed absolute k — use case
  sizes vary 4x), matching the actual usage pattern (a ranked list a human reviews).
- **Secondary: PR-AUC / Average Precision** — more informative than ROC-AUC given how
  much class balance varies by use case (`solar_leo` 77% positive, `soil_microbiome`
  26% positive).
- **F2** if a hard threshold decision is ever needed (recall weighted over precision).
- **ROC-AUC kept for continuity** with everything already measured this way.
- **Calibration (Brier score / reliability curve) per use case** — matters more now
  that `relevance_score` can't be leaned on for an intuitive "is 0.7 high" reading; a
  model's own probabilities need to carry that meaning instead.
- On SYNERGY specifically: given its ~1.7% inclusion rate, lean on Recall@k and PR-AUC
  there even more than on the in-repo use cases — ROC-AUC is easy to inflate under that
  much imbalance.

---

## 3. Ensemble/stacking architecture

Revised from the previous recommendation with `relevance_score` removed as a feature
candidate — which makes the **metadata-only, embedding-free arm more important**, not
less: with the one candidate "portable" signal disqualified, whether a robust,
version-safe structured signal exists at all (term-overlap, citation-based features,
source booleans) is now an open question this arm needs to actually test, not something
assumed already answered.

| Component | Recommendation |
|---|---|
| Base learner A | Tree-based (HistGradientBoosting or LightGBM/CatBoost) on raw embeddings + engineered metadata — already validated at 0.832 in-distribution |
| Base learner B | Linear model (LogisticRegression) on PCA-reduced embeddings + metadata — already validated at 0.808, K=30 |
| Base learner C (new) | **Metadata-only, no embeddings at all** — term-overlap, `citation_velocity`, per-use-case percentile ranks, source booleans, `has_abstract`. This is now the ensemble's main hope for a cheap, portable, version-safe signal, and needs to be evaluated via leave-one-use-case-out on its own merits, not assumed to inherit `relevance_score`'s (retracted) generalization story |
| Base learner D (optional, diversity) | kNN on embeddings, or a different tree library — decorrelates errors from A without a new inductive bias story |
| Combiner | Weighted/learned average as the floor (already shown to match a plain meta-learner); a `use_case_key`-aware gating meta-learner as the one worth testing beyond that, so blending weights can differ by domain rather than being fixed |
| **Selection criterion** | **Leave-one-use-case-out ROC-AUC/PR-AUC, stratified by `embed_model`** once that varies — not in-distribution CV. Reject any architecture that wins in-distribution and doesn't move the held-out number, same principle as before, now with the added check that the win isn't an `embed_model`-specific artifact |

---

## 4. Global model + per-use-case fine-tuning — viable, scoped by label count

The "~50 informative labels + well-defined use case should bound 'interesting' for
recall" estimate changes this from an open question into a concrete design constraint:
50 labels is a **warm-start calibration budget, not a hyperparameter-search budget.**
Grid search (or any per-use-case architecture search) at n≈50 will overfit to noise —
this was already a concern with the existing use cases' few-hundred-row folds; at 50
it's not a risk, it's close to a guarantee.

**Recommended adaptation ladder, keyed to how many labels a use case actually has:**

| Labels available | What's viable | What isn't |
|---|---|---|
| 0 (brand-new use case, no labels yet) | Unsupervised ranking only — embeddings + term-overlap similarity, frozen shared model | Any learned adaptation |
| ~50 (the stated bootstrap target) | **Probability/threshold calibration only** — isotonic regression or Platt scaling on top of the frozen shared model's output, to set the recall-bound decision point for that use case | Hyperparameter search, retraining, anything with more than a handful of fitted parameters |
| Low hundreds (where the 6 existing use cases already sit) | Light per-use-case adjustment — the gating meta-learner's blending weights, or a small per-use-case correction layer | Full independent retraining per use case, aggressive multi-parameter grid search |
| Large, stable pool | Full nested-CV hyperparameter search becomes defensible | — |

This directly answers the question as posed: **"global model, then grid-search
fine-tune per use case" is viable for the existing 6 use cases (low hundreds of
labels each), but not as the design for a genuinely new use case at the ~50-label stage
that's actually being planned for.** Build the calibration-only path as the real target
for new use cases, and treat grid-search fine-tuning as something that only turns on
once a use case's label count crosses into the hundreds — verify that crossover
empirically per use case rather than assuming a fixed number.

---

## 5. Search strategy (grid search alternatives)

Unchanged in substance from the previous recommendation, now reinforced by §4's numbers
rather than a general small-sample worry:

- **Bayesian optimization (Optuna) over grid search** wherever more than 2–3
  hyperparameters are in play — finds comparable results in far fewer trials, which
  matters because trials here are expensive in *variance*, not compute.
- **Nested CV always**, for the low-hundreds regime where hyperparameter search is
  viable at all (§4) — hyperparameters picked on the same folds you report final
  numbers on will look better than they are.
- **Repeat CV across multiple random seeds** and average, given how small and
  group-constrained the dev pool is (1,588 rows across 5 use cases).
- **Leave-one-use-case-out as the outer loop** for any architecture or hyperparameter
  decision, not a single fixed holdout.
- **Successive halving** (`HalvingRandomSearchCV`) if throwing more compute at search —
  explores more candidates per unit of wall-clock than a blind grid.
- The standing contrarian point: most of the "more sophisticated" things tried so far
  didn't beat the simple baseline (stacking ≈ averaging, spaCy ≈ plain regex). Spend
  search budget on the generalization axis (leave-one-use-case-out, cross-`embed_model`
  robustness), not on squeezing in-distribution fit — that's not the demonstrated
  bottleneck.

---

## 6. Generalize vs. specialize — now an open question again, not a settled one

The previous recommendation leaned on `relevance_score`'s apparent portability as
evidence that "cheap structured signals transfer better than embeddings across
domains." With that evidence retracted (§0), **this is now genuinely unresolved** — not
wrong, just unproven. Term-overlap is the only remaining structured-feature candidate
with any tested signal, and it's use-case-conditional by construction (real on 3 of 6
use cases, dead on the other 3), which isn't the same claim as "transfers well to a new
domain."

**Priority action before committing to an architecture:** re-run the
generalize-vs-specialize comparison properly — evaluate base learner C (§3, the
metadata-only arm) via leave-one-use-case-out specifically, the same discipline already
applied to the embedding baseline, and see whether it actually holds up out-of-domain.
Don't assume it will just because `relevance_score` looked like it would.

**Once that's re-tested, the architecture shape I'd still expect to win** (unless the
re-test says otherwise): a shared backbone (embeddings + whatever in §3 actually proves
portable) plus the label-count-aware adaptation ladder from §4 — not a monolithic global
model, not independent per-use-case models. For a genuinely new use case: unsupervised
ranking at 0 labels, calibration-only at ~50, light adaptation once labels grow — the
same ladder, just no longer justified by a retracted finding.

---

## 7. Open action items

1. Confirm `uc1`'s availability and pin an export of it for clean evaluation.
2. Pull in the SYNERGY dataset as an independent validation benchmark.
3. Confirm when/whether `use_case_version` appears in fresh exports; add it to
   `combine_use_cases.ipynb`'s broadcast step once it does.
4. Re-test whether any structured/metadata-only feature set actually generalizes
   cross-domain (leave-one-use-case-out) — the open question left by §6, now that
   `relevance_score` no longer answers it.
5. Design the ~50-label calibration step (isotonic/Platt) as the real target for new
   use cases, not a placeholder for eventual grid search.
6. Treat every number compared against a past result as version-pinned — this applies
   beyond `relevance_score`; any live-refined artifact from `academic_research_agent`
   should be assumed unversioned until proven otherwise.
