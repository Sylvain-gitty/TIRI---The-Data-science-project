# TIRI — The Data Science Project

[![CI](https://github.com/Sylvain-gitty/TIRI---The-Data-science-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Sylvain-gitty/TIRI---The-Data-science-project/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

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
| **External, at realistic prevalence** | How well does any of it do where only **2%** of papers are relevant, on questions nobody here labelled? | Per-silo ensemble **ROC-AUC 0.891**; a prompted LLM reaches 0.834 and **does not** replace it |

### Pooled — the model comparison (`notebooks/main/06`–`08`)

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

### Within-question — the production surface (`notebooks/main/09`, `reports/`)

One model per research question, fit only on that question's own labels. Per-silo
CatBoost + LogisticRegression, 50/50 averaged. Validated externally against three
[SYNERGY](https://github.com/asreview/synergy-dataset) systematic reviews we had no hand
in labelling, at realistic prevalence (1.7–14.8% positive, vs 26–77% in our own pools):
**mean ROC-AUC 0.899**, against 0.888 for the best single-embedding baseline ever
measured here.

The full architecture decision doc — every choice, its evidence, and its confidence
grade — was `reports/wf_ensemble_final_recommendations.md` (see **Removed reports** below).

### Can a prompted LLM do this instead? (`notebooks/main/11`, `reports/wf_llm_*`)

Short answer: **no, and the interesting part is why not.**

Tested on an external 28-collection benchmark — 8 collections, 62,229 papers, **2.19%
relevant**, which is the first surface here where F2 discriminates at all (at our own
pools' 26–77% prevalence, "mark everything relevant" already scores F2 0.872, so nothing
could be told apart). Four open-weights models from 20B to 397B, four prompt variants,
89,937 responses, $16.10.

| labels per question | method | mean ROC-AUC |
|---|---|---|
| 0 | cosine-to-brief | 0.774 |
| 0 | best prompted LLM | 0.815 |
| 60 | LLM + a brief **induced from those 60 labels** | **0.834** |
| 60 | LogReg on the embedding | 0.844 |
| in-silo | **per-silo ensemble** | **0.891** |

Zero-shot, every model beat the free cosine baseline on exactly **3 of 8 collections**
against a pre-registered ≥6/8. Two of this repo's earlier conclusions did not survive the
move to realistic prevalence — the model ranking scrambled completely, and the
prompt-engineering trick that was worth +0.35 F2 at high prevalence is worth ±0.03 here.

What did work is a **better brief**, not a better model: showing a strong model 30 relevant
and 30 irrelevant papers and asking it to write the screening rule it infers is worth
**+0.057 ROC-AUC** for one $0.006 call — larger than the entire 20B→397B model spread
(0.030). It is a **cold-start lever only**: folded into the trained ensemble it is worth
**+0.001**, because the embedding block already encodes it.

Decision doc: `reports/wf_llm_benchset_a_findings.md`. Every method and metric side by
side: `reports/wf_llm_benchset_a_summary.md`. Both were removed — see **Removed reports**
below.

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

`requirements.txt` carries documented version *floors*, which is what you want for
development. To reproduce the exact environment the main line was last verified in — pandas
pinned, everything pinned — use [`requirements-lock.txt`](requirements-lock.txt) instead;
its header says precisely what that set is and is not certified to reproduce.

**The data you need is in the repo.** Clone and run — no API keys, no GPU, no export
step. Notebooks assume they're run with their own folder as the working directory.

| Runs from a fresh clone | Needs regenerating first |
|---|---|
| `main/01_data_compile` — rebuilds the cleaned corpus from raw | `main/04_feature_engineering` — needs `embeddings_cache/` (~295 MB) |
| `main/02_eda_quickstart`, `main/03_eda_full` | `main/06`–`08` — need the full `papers_fe.parquet` (106 MB) |
| `main/05_validation_design` — where the collapse is measured | `main/09_ensemble_per_silo` — same, plus Modal |
| **16 of the 18** `notebooks/experiments/` — the evidence | `experiments/wf_synergy_validation` — needs an OpenRouter key in `.env` |
| both `notebooks/future_work/` templates (gated, safe to Run All) | `experiments/wf_top_embeddings_generalization` — needs `embeddings_cache/` |
| | the four `*_benchset_v1` notebooks — need `data/benchsets_v1/` (92 MB, not tracked; see **Data** below), and 02/03/04 also need `scripts/embed_benchsets.py` to have run |
| | `experiments/sf_ensemble_benchset_v1_large_set_a/b_plus_papers_fe` — need Modal (a dedicated app/volume) plus the ungit-tracked `benchset_v1_*` files; both executed, with real results committed |

The left-hand column is **verified, not asserted**. `git clone` into an empty directory,
`pip install -r requirements.txt`, Run All on every notebook: the main line passes — 01
(3s), 02 (5s), 03 (37s), 05 (124s) — and 16 of the 18 supporting notebooks pass. `01`
reproduces the committed `papers_combined.parquet` frame-for-frame, so the cleaning step
is checkable rather than trusted.

Re-verified on **Windows, Python 3.11.3, pandas 3.0.5** (2026-08-19): 01, 02, 03 and 05 all
run clean, and 01 still reproduces `papers_combined.parquet` frame-for-frame — 2,873 × 50,
identical dtypes and values. Worth stating because the first pass was macOS-only, and it
turned out that `01` did **not** run on Windows: text I/O inherited the platform's default
encoding, so the three briefs containing an en-dash failed to read. That is fixed
throughout, and CI now guards the class.

**Install everything, not just the light half.** `fastembed` and `sentence-transformers`
(which pulls PyTorch) are `requirements.txt`'s two heaviest entries and the easiest to
skip — but anything that embeds text from scratch needs them, and the failure is a bare
`ModuleNotFoundError` several cells in rather than at import time.

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
  README.md         the index — what every notebook does and what it found
  main/             the main line, numbered in reading order
                      01 data compile   02 EDA quickstart   03 EDA full
                      04 feature eng.   05 validation design
                      06 baseline       07 CatBoost         08 ensemble (pooled)
                      09 ensemble (per-question — the production surface)
                      11 LLM screening on the external benchmark, at 2% prevalence
  experiments/      supporting evidence: feature viability checks, embedding
                    bake-offs, external validation, superseded passes. Mostly
                    negative results, kept on purpose. Never writes to data/

scripts/            shared libraries (embedding, folds, lexical features, metrics) + experiment drivers
tests/              invariants of the shared libraries
reports/            empty. Notebooks and scripts write their outputs here; the
                    written-up decision trail was removed (see below)
```

Only `main/01` and `main/04` write to `data/`. Everything else is read-only, so
notebooks can be run in any order once the data exists.

### Where to start reading

1. This file, then [`CONTEXT.md`](CONTEXT.md) for the findings that shaped the work.
2. [`main/02_eda_quickstart.ipynb`](notebooks/main/02_eda_quickstart.ipynb) — the
   5-minute tour of the corpus.
3. [`main/03_eda_full.ipynb`](notebooks/main/03_eda_full.ipynb) — the full EDA pass,
   with hypothesis tests.
4. [`main/05_validation_design.ipynb`](notebooks/main/05_validation_design.ipynb)
   — the validation design, and where the generalisation collapse is measured.
5. [`main/06_baseline_logreg.ipynb`](notebooks/main/06_baseline_logreg.ipynb)
   → [`07_advanced_catboost.ipynb`](notebooks/main/07_advanced_catboost.ipynb)
   → [`08_ensemble_pooled.ipynb`](notebooks/main/08_ensemble_pooled.ipynb) — baseline,
   advanced model, ensemble, in that order. Each is self-contained and reports train →
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
| `papers_fe.parquet` | 106 MB | `notebooks/main/04_feature_engineering.ipynb` |
| `papers_fe_synergy*.parquet` | 173 MB | `scripts/run_synergy_recall_validation.py` |
| `embeddings_cache/` | 295 MB | `scripts/modal_embeddings.py` (GPU) + OpenRouter (paid) |
| `data/benchsets_v1/` | 92 MB | re-download from the sources in its own `README.md` (all CC0 / CC BY) |
| `papers_benchset_v1.parquet` | 157 MB | `notebooks/main/01_data_compile_benchset_v1.ipynb` |
| `benchset_v1_{small_test,large_set_a,large_set_b}.parquet` | 2.7 GB total | `notebooks/main/04_feature_engineering_benchset_v1.ipynb` |

### The benchmark corpus

`data/benchsets_v1/` is a second, externally-sourced corpus: **28 published
systematic-review screening collections, 181,199 papers, 1.86% relevant**, with real
expert labels and LLM-drafted (label-blind) briefs. It exists because the six questions
above run **26–77% positive** — roughly 20× production prevalence — so every F2 number and
calibrated threshold measured on them was measured in the wrong regime (`CONTEXT.md` §3).
This corpus is the prevalence-realistic surface.

It is not tracked in git: 92 MB is four times everything else here, and it isn't ours to
redistribute. It has its own `README.md` naming each source. Four notebooks consume it —
`01_data_compile_benchset_v1`, `02_eda_quickstart_benchset_v1`, `03_eda_full_benchset_v1`,
`04_feature_engineering_benchset_v1` — and `scripts/embed_benchsets.py` computes the
Jasper + Qwen3-4B vectors they join. **Run the script first.**

**The three benchmark sets.** 04 turns the corpus into `benchset_v1_small_test.parquet`
(13 collections, evaluation only — too few positives to cross-validate, or not a screening
task) plus `benchset_v1_large_set_a` / `_set_b` (8 and 7 collections, 62k and 82k rows).
A and B are **development vs held-out, both used within-silo** — not a cross-silo
train/test split, which `CONTEXT.md` §1 rules out. The partition is the exact minimum-leak
balanced split of all 16,384 possibilities; `reports/benchset_v1_split_manifest.json`
records how it was chosen and what it costs.

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
2. Add a `USE_CASE_REGISTRY` entry in `notebooks/main/01_data_compile.ipynb`
   (it raises loudly on an unregistered question rather than guessing a short code).
3. Re-run that notebook, then feature engineering.

The registry check is deliberate: a new question needs a considered short code, because
that code is what every downstream report will call it.

---

## Future work

**Continued experimentation with the advanced model** — the near-term list, in priority
order (full reasoning was in `reports/wf_ensemble_final_recommendations.md`):

- **Replace the ~6,000-dim embedding block with a single supervised discriminant
  direction.** Scored statistically identical (F2 0.897 vs 0.892) at a fraction of the
  compute. A real simplification, needs SYNERGY validation before adoption.
- **Re-run the feature ablation under the now-tuned CatBoost.** A stronger base learner
  can absorb what weaker engineered features were compensating for; never re-checked
  after tuning landed.
- **A per-question lean model for `cement_binders`.** A lexical+metadata-only CatBoost
  (no embedding) beat *both* production branches standalone there (ROC-AUC 0.933 vs
  0.912/0.876). It's the one question where dropping the embedding wins.

**From the LLM screening work** (full list was in `reports/wf_llm_benchset_a_findings.md`
§10):

- **Run the induced brief on the collections that are too small to train on.** 12 of the
  benchmark's 28 collections are too small to split, which is exactly the cold-start
  population the induced brief serves — and the one the run above could not test.
- **Locate the crossover.** A better brief is worth +0.057 ROC-AUC at 60 labels and +0.001
  once a question has enough to train on. Only those two points were measured, so the label
  count where it stops paying is unknown — and it is the number that decides when to stop
  paying for one. Free: no new inference needed.
- **An induced brief vs few-shot examples on the same 60 labels.** Decides whether the
  artefact should be a brief or a set of demonstrations. A brief is far more attractive
  operationally: human-readable, auditable, editable by an analyst, and it also feeds the
  non-LLM lexical features, which in-context examples cannot.
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


---

## Removed reports

`reports/` held 57 written-up reports — the decision trail behind every number above,
including the two architecture decision docs this README cites. They were removed to slim
the working tree and the directory is now empty except for a `.gitkeep`, because notebooks
`02` and `03` write their figures there.

**Nothing is lost.** Every one is in git history and recoverable individually:

```bash
git show 75c14de:reports/wf_ensemble_final_recommendations.md
git checkout 75c14de -- reports/          # restore the whole folder
```

The consequence worth stating plainly: claims in this README and in `CONTEXT.md` now cite
evidence that is not in the working tree. The measurements behind them were not revised —
the write-ups were removed — but a reader taking this repository at face value can no
longer check them without going to history.

Citation metadata is in [`CITATION.cff`](CITATION.cff). A negative result, clearly
measured, is a welcome contribution — see [`CONTEXT.md`](CONTEXT.md) §6, the register of
everything already tried and rejected.

## Licence

**Not yet licensed.** No licence file has been chosen, which means default copyright
applies and the code and data here are **not** yet reusable by others, whatever this
repository's visibility. A licence needs to be added before that changes.

Note also that `data/` redistributes third-party academic metadata (Crossref, OpenAlex,
arXiv, PubMed, Europe PMC, CORE, Semantic Scholar) alongside this project's own analyst
labels. The terms attaching to that metadata are a separate question from the licence on
the code, and both need settling before anyone should treat this repository as
open source.
