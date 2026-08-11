# TIRI — The Data Science Project

Turning a labelled corpus of academic literature into a working relevance-screening
model: clean → feature-engineer → validate → model → ensemble → evaluate.

A four-week capstone by two people, built on real data from a real product. The
interesting part is not the accuracy number — it's the finding that the obvious
approach was measurably the wrong one, and what we did about it.

---

## The problem

A researcher defines a question ("find alternative cement binder chemistries that cut
embodied carbon by more than half"), a search returns a few hundred candidate papers,
and a human has to read all of them. We want a model that ranks the pile so the
relevant papers surface first — **high recall first**, precision later.

TIRI does not scrape or label anything. That's the job of a sibling tool,
`academic_research_agent`, which runs the search → triage → label loop and exports
one row per paper: title, abstract, metadata, the analyst's `triage_label`, a
`relevance_score`, and a precomputed embedding. TIRI consumes that export.

**Six research questions, 2,873 papers, 1,852 with a usable binary label** (a further
543 were marked "borderline" and 478 were never reviewed — see
[`data/processed/README.md`](data/processed/README.md) for why those are three different
facts, not one).

## The central finding

The obvious move is to embed each paper and train a classifier. We did that, and it
scored well — right up until we tested it on a research question it hadn't seen.

> **Relevance is a property of the (brief, paper) *pair*, not of the paper.**

Every feature we built at first described the paper alone. The measured consequences:

- A LogisticRegression predicts **which research question a paper belongs to from its
  embedding alone, at 96.2% accuracy** (majority class 19%). The embedding isn't noisy —
  it's a *fingerprint of the question*, not of relevance.
- So a model trained across questions learns which question it's looking at, which is
  definitionally non-transferable. Leave-one-question-out performance collapses to
  **~0.54 ROC-AUC** — a coin flip.
- This is **not** "too many features". Ten *brief-relative* scalars beat the full
  384-dimension embedding on the same test (**0.642 vs 0.537**). It was never the
  feature count; it was the feature *kind*.
- Strongest form: with **zero** labels, ranking papers by plain cosine similarity to the
  brief beats every supervised cross-question model we trained (ROC-AUC **0.693 vs
  0.610**). A model trained on other people's questions is worse than no model at all.

That reframed the project. Features must read the brief, and models are fit **per
research question, never pooled**. Full technical detail — including everything we
tried that didn't work — is in [`CONTEXT.md`](CONTEXT.md).

---

## Results

**Read the surface column first.** These numbers are not comparable to each other, and
which one matters depends on the question being asked. Conflating them is the single
easiest way to misread this project.

| Surface | What it answers | Headline |
|---|---|---|
| **Pooled** | How well does a model do on *more papers from questions it has already seen*? | Ensemble holdout **ROC-AUC 0.889** |
| **Leave-one-question-out (LOGO)** | How well does it do on a *brand-new question* with no labels? | **~0.54 ROC-AUC — it doesn't.** This is the finding, not a failure to fix |
| **Within-question (per-silo)** | How well does it do on a *new paper for a question we have labels for*? — **the production surface** | Per-silo ensemble clears the strong baseline by **0.03–0.18 ROC-AUC** across the six questions |

### Pooled — the model comparison (`notebooks/modelling/`)

A random holdout drawn from the same six questions. This is the standard bootcamp
comparison, and it is *explicitly not* a generalisation test.

| Model | Holdout ROC-AUC | F2 @ 0.5 | F2 @ tuned threshold | Note |
|---|---|---|---|---|
| LogisticRegression baseline | 0.867 | 0.790 | — | `StandardScaler` → `PCA(50, whiten)` on the embedding block (58.0% variance retained) |
| CatBoost (tuned for F2) | 0.864 | 0.798 | 0.887 @ 0.2 | Train→validation AUC 0.997→0.882 — overfits ~3x harder than the baseline |
| **Stacked ensemble** | **0.889** | 0.810 | **0.892 @ 0.2** | CatBoost + LogReg + RandomForest, LogisticRegression meta-learner on out-of-fold predictions |

The F2 columns are split deliberately: at the default 0.5 threshold the three models sit
within 0.02 of each other, and most of the apparent F2 spread is a **decision-threshold
effect, not a ranking-quality difference**. CatBoost's AUC and average precision are
identical at 0.5 and 0.2 (0.864 / 0.897) — only recall moves (0.793 → 0.934). ROC-AUC is
the fairer lens on which model actually ranks better.

Every comparative claim here is backed by a **paired bootstrap test on the holdout**,
not a point-estimate table. What that discipline bought us:

- The ensemble beats the baseline significantly on both AUC (+0.022, p=0.005) and F2
  (+0.101, p<0.001).
- It beats standalone CatBoost on AUC (+0.025, p<0.001) — a genuine ranking-quality gain
  from stacking itself.
- But its F2 edge over standalone CatBoost (+0.005) is **not significant** (p=0.596, CI
  crosses zero). The point-estimate table alone would have let us claim a win there.

Two honest caveats we'd rather state than bury:

- **Judged fairly, the "advanced" models barely beat the simple one.** At each model's
  own best threshold, out-of-fold F2 is LogReg 0.903, CatBoost 0.902, RandomForest 0.898
  — a spread of 0.005. The ensemble's real contribution (0.910) comes from *stacking*,
  not from any base model being better.
- **That +0.101 F2 compares a threshold-tuned ensemble against an untuned baseline.**
  Both thresholds were tuned on training-fold out-of-fold predictions and never on the
  holdout, so the number is not leaked — but the baseline was never given the same
  treatment, and on the like-for-like 0.5 comparison the gap is +0.020, not +0.101.

### Within-question — the production surface (`notebooks/pipelines/`, `reports/`)

One model per research question, fit only on that question's own labels. Per-silo
CatBoost + LogisticRegression, 50/50 averaged. Validated externally against three
[SYNERGY](https://github.com/asreview/synergy-dataset) systematic reviews we had no hand
in labelling, at realistic prevalence (1.7–14.8% positive, vs 26–77% in our own pools):
**mean ROC-AUC 0.899**, against 0.888 for the best single-embedding baseline ever
measured here.

The full architecture decision doc — every choice, its evidence, and its confidence
grade — is [`reports/wf_ensemble_final_recommendations.md`](reports/wf_ensemble_final_recommendations.md).

### A note on what counts as a result here

Measured seed-to-seed noise is **~0.010 ROC-AUC** and 0.015–0.027 WSS@95. We treat any
gap below **~0.03 as not established**, regardless of which direction it points. Several
things in this repo that look like wins are labelled as ties for exactly that reason.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # only for the NER comparison script
```

**The data you need is in the repo.** Clone and run — no API keys, no GPU, no export
step. Notebooks assume they're run with their own folder as the working directory.

| Runs from a fresh clone | Needs regenerating first |
|---|---|
| `notebooks/data_compile/` — rebuilds the cleaned corpus from raw | `notebooks/feature_engineering/` — needs `embeddings_cache/` (~295 MB) |
| `notebooks/eda/` — EDA, label balance, embedding maps | `notebooks/modelling/sf-*` — need the full `papers_fe.parquet` (106 MB) |
| `notebooks/modelling/sf_logo_fold_strategy.ipynb` — the validation design | `notebooks/pipelines/wf_ensemble_fold_pipeline.ipynb` — same, plus Modal |
| `notebooks/feature_experiments/` — the negative-results evidence | |

Everything in the right-hand column is **committed with its outputs intact**, so the
results are readable without re-running. See [Data](#data) for exactly what ships and
how to rebuild the rest.

---

## Repo map

```
CONTEXT.md          the technical findings + negative-results register — read this second
README.md           you are here

data/
  raw/              6 labelled exports (<question>.jsonl) + their briefs (<question>.usecase.json)
  processed/        papers_combined.parquet, papers_fe_slim.parquet, and README.md (data dictionary)

notebooks/
  README.md         one-line description of every notebook — the index
  data_compile/     raw exports -> papers_combined.parquet
  eda/              exploratory analysis, read-only
  feature_experiments/   viability checks for candidate features — mostly negative results, kept on purpose
  feature_engineering/   assembles validated features -> papers_fe.parquet
  modelling/        fold/validation design, and the final baseline + advanced models
  pipelines/        config-driven fold pipelines, incl. the per-silo ensemble
  comparisons/      runs the compare_*.py scripts with output inline

scripts/            shared libraries (embedding, folds, lexical features, metrics) + experiment drivers
future_work/        parked work, kept but not wired into the current workflow
reports/            the decision trail — what we tried, what we measured, what we rejected
```

### Where to start reading

1. This file, then [`CONTEXT.md`](CONTEXT.md) for the findings that shaped the work.
2. [`notebooks/eda/eda_quickstart.ipynb`](notebooks/eda/eda_quickstart.ipynb) — the
   5-minute tour of the corpus.
3. [`notebooks/eda/sf_eda_v2.ipynb`](notebooks/eda/sf_eda_v2.ipynb) — the full EDA pass,
   with hypothesis tests.
4. [`notebooks/modelling/sf_logo_fold_strategy.ipynb`](notebooks/modelling/sf_logo_fold_strategy.ipynb)
   — the validation design, and where the generalisation collapse is measured.
5. [`notebooks/modelling/sf-LogRegbaseline.ipynb`](notebooks/modelling/sf-LogRegbaseline.ipynb)
   → [`sf-catboost-model.ipynb`](notebooks/modelling/sf-catboost-model.ipynb)
   → [`sf-ensemble.ipynb`](notebooks/modelling/sf-ensemble.ipynb) — baseline, advanced
   model, ensemble, in that order. Each is self-contained and reports train →
   validation → holdout at every stage.

---

## Data

`data/` is one pair of files per research question, named by its short key:

```
data/raw/<question_key>.jsonl          the labelled export (papers + labels)
data/raw/<question_key>.usecase.json   the brief (objective, must/nice/exclude terms, decision rules)
```

The six keys are `carbon_capture`, `cement_binders`, `ner`, `soil_microbiome`,
`solar_leo`, `tech_forecasting` — the same keys used in every notebook and report.

### What ships in git, and why

The default rule is that data doesn't belong in git. We broke it deliberately: a repo
whose every notebook dies on cell 1 with `FileNotFoundError` can't be reviewed. So we
ship whatever is small enough **and** needed to run the main line (~22 MB), and ignore
the bulk that's regenerable (~600 MB).

| Tracked | Size | What it is |
|---|---|---|
| `data/raw/*.jsonl` | 27 MB (12.5 MB packed) | The six labelled exports — source of truth |
| `data/raw/*.usecase.json` | ~2 KB each | The briefs |
| `data/processed/papers_combined.parquet` | 9.3 MB | Cleaned corpus, 2,873 × 50, incl. the original embedding |
| `data/processed/papers_fe_slim.parquet` | 0.3 MB | Feature table, 1,848 × 38 — everything except the embedding blocks |

| Ignored | Size | Rebuild with |
|---|---|---|
| `papers_fe.parquet` | 106 MB | `notebooks/feature_engineering/wf_build_fe_dataset.ipynb` |
| `papers_fe_synergy*.parquet` | 173 MB | `scripts/run_synergy_recall_validation.py` |
| `embeddings_cache/` | 295 MB | `scripts/modal_embeddings.py` (GPU) + OpenRouter (paid) |

**Why a "slim" feature table.** `papers_fe.parquet` is 1,848 × 8,742 and 106 MB — 101 MB
of it three raw embedding blocks. Drop those and 38 columns weighing 0.3 MB remain: the
labels, the fold grouping key, the query-conditioned lexical block, cosine-to-brief, and
metadata. That's enough to reproduce the **central finding above** with no embedding
model, no API key and no GPU. Regenerate it with:

```bash
python scripts/export_slim_fe.py
```

Column-by-column documentation for both processed files is in
[`data/processed/README.md`](data/processed/README.md).

### Adding a seventh research question

The pipeline discovers files by glob and matches them on the `use_case` name recorded
*inside* each file, not on the filename — so adding data is:

1. Drop `<new_key>.jsonl` and `<new_key>.usecase.json` into `data/raw/`.
2. Add a `USE_CASE_REGISTRY` entry in `notebooks/data_compile/combine_use_cases.ipynb`
   (it raises loudly on an unregistered question rather than guessing a short code).
3. Re-run that notebook, then feature engineering.

The registry check is deliberate: a new question needs a considered short code, because
that code is what every downstream report will call it.

---

## Future work

**Continued experimentation with the advanced model** — the near-term list, in priority
order (full reasoning in
[`reports/wf_ensemble_final_recommendations.md`](reports/wf_ensemble_final_recommendations.md)):

- **Replace the ~6,000-dim embedding block with a single supervised discriminant
  direction.** Scored statistically identical (F2 0.897 vs 0.892) at a fraction of the
  compute. A real simplification, needs SYNERGY validation before adoption.
- **Re-run the feature ablation under the now-tuned CatBoost.** A stronger base learner
  can absorb what weaker engineered features were compensating for; never re-checked
  after tuning landed.
- **A per-question lean model for `cement_binders`.** A lexical+metadata-only CatBoost
  (no embedding) beat *both* production branches standalone there (ROC-AUC 0.933 vs
  0.912/0.876). It's the one question where dropping the embedding wins.
- **A proper central hyperparameter search** via the LOGO harness, rather than the
  two-point sweep used so far — searching once centrally is the rule, per `CONTEXT.md` §1.
- **Threshold/calibration transfer.** Every F2 number here was measured at ~20x
  production prevalence. Recall@k is the right primary metric; a fixed threshold is not.

**More data for testing** — the current evidence base is six questions of 260–360
labelled rows, which is small and heterogeneous enough that we treat sub-0.03
differences as noise:

- **23 unused SYNERGY reviews are ring-fenced.** Many decisions here were made against
  LOGO scores, so LOGO is no longer an unbiased surface. Those 23 reviews are the only
  clean evaluation surface left — do not spend them on model selection.
- **A seventh and eighth research question** would do more for confidence than any
  further tuning. Several findings currently rest on 3-of-6 or 4-of-6 splits.
- **Label recall is unmeasured.** One labeller means no inter-annotator agreement, and a
  labeller's false negative is indistinguishable from a true negative — it silently
  inflates measured recall. Fix: sample the negatives and re-label blind.
- **Prevalence-realistic pools.** Ours run 26–77% positive; production is low single
  digits. Recall metrics barely discriminate when there's no room to skip.

`CONTEXT.md` §4 carries the full open-risk register, including the ones we could not
close.

---

## How we work

### Non-negotiable conventions

- **NULL is not 0.** "We never found out" and "we looked and there was nothing" are
  different facts. Features over sparse fields emit NaN plus an indicator, rather than
  silently filling zero.
- **Cross-validated, never train-then-score.** Any accuracy/AUC/F2 in this repo is on
  held-out folds. A score on data the model was fit on is not a result.
- **Label which critic is speaking.** Where a hard metric sits next to a judgement call,
  say which is which — a reader can't otherwise tell a measurement from an opinion.
  Notebooks carry explicit "Finding" cells for this.
- **Anything claiming to read the brief must survive a falsification control.** Rebuild
  the feature against deliberately *wrong* briefs; if it still scores well, it's
  measuring something generic and gets thrown away. This caught more than one plausible
  feature.

### Collaboration

Branches are prefixed with the author's initials (`wf-`, `sf-`), one logical change per
commit, and everything lands on `main` through a reviewed pull request. Notebook
filenames carry the same prefix, so the two work streams stay legible in a shared repo.

**Contributions.** SF: the EDA passes, the LOGO fold strategy, and the baseline →
CatBoost → ensemble modelling line. WF: data compilation, the feature-engineering track
(query-conditioned lexical features, embedding bake-off), the per-silo ensemble, and
external validation on SYNERGY.
