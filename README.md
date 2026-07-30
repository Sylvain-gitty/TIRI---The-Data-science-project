# TIRI — The Data Science Project

TIRI is the data-science working repo for turning a labelled academic-literature corpus
into a real, end-to-end ML workflow: cleaning → feature engineering → modelling →
ensembling → evaluation. It's built as a month-long project, structured week by week
below.

## Where the data comes from

TIRI does not scrape or label papers itself — that's the job of the sibling repo,
[`academic_research_agent`](../academic_research_agent). That app turns a use case into
a search → triage → label loop and can **export an ML-ready parquet**: one row per paper,
with title/abstract/metadata, the analyst's `triage_label` and `review_label`
(positive / negative / pass), a `relevance_score`, and a precomputed sentence
**embedding** + the `embed_model` that produced it.

TIRI takes that export as its raw input. This split matters: labelling is a slow,
human-in-the-loop process (that's what the agent is for); once a corpus is labelled,
everything from here is standard data science and can iterate fast without touching the
labelling tool again.

To get data into TIRI: run an export from the agent's Data Analytics screen (or point it
at an existing export you already have) and drop the parquet file into `data/raw/`.

## Project workflow (4 weeks)

**Week 1 — Ingest & clean**
Load the export, understand its shape (`data/raw/*.parquet` → `data/processed/`).
Handle nulls honestly (a `NULL` field means "we never found out", not "zero" — carried
over from the agent's own convention). Dedupe, normalise text (strip markup from
abstracts), check label balance, EDA on year/venue/citation_count/language/source.

**Week 2 — Feature engineering**
Text features: the embeddings that ship with the export, plus alternatives (see
`scripts/compare_embeddings.py` below — this is the week to run it, inspect the latent
space of each candidate model, and pick one). Metadata features: citation counts,
venue/source one-hots, recency.

**Week 3 — Modelling**
Baselines first (logistic regression on embeddings — this is exactly what the agent's own
`model.py` classifier does for triage re-ranking, so it's a fair floor to beat). Then
ensembles: random forest, gradient boosting, stacking. Cross-validate honestly
(`StratifiedKFold`, no leakage between folds) — a score on data the model was trained on
is not a result.

**Week 4 — Evaluate, error-analyse, write up**
Held-out evaluation, error analysis (which papers does the model get wrong, and why),
compare against the agent's own relevance ranking as a baseline, final report.

## Repo layout

```
data/
  raw/          exports dropped in as-is (gitignored — data doesn't belong in git)
  processed/    cleaned / feature-engineered outputs of your own pipeline
notebooks/      exploratory notebooks (one per week/topic is fine)
scripts/        standalone, runnable analysis scripts
  compare_embeddings.py   week-2 embedding model comparison (see below)
reports/        write-ups, figures, model comparison tables
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Scripts

### `scripts/compare_embeddings.py`

Visually compares the **latent space** of several embedding/NLP models on your labelled
export — a lexical TF-IDF baseline plus one or more fastembed neural models by default.
For each model it draws a row of three plots:

1. a 2D map of the whole corpus (PCA or t-SNE), coloured by triage label, with the
   corpus centroid and the embedded use case marked;
2. a pairwise-cosine-similarity histogram — is this model's latent space **dense**
   (every paper's vector points the same way — a known failure mode, "anisotropy"/
   representation collapse) or **disperse** (vectors actually spread out by content)?
3. a histogram of how similar each abstract is to the corpus centroid, with the use
   case's own similarity marked — **how central is the use case relative to its own
   corpus?**

It also reports cross-validated ROC-AUC (same lens as the agent's own adaptive triage
classifier, `model.py`) as a secondary "does this space support classification" check.

```bash
python scripts/compare_embeddings.py --data data/raw/your-export.parquet
```

Outputs: `reports/latent_space_comparison.png` (the visual comparison) and
`reports/embedding_comparison.csv` (the scalar metrics behind it). See the script's own
docstring for the full walkthrough — what each metric means, why those specific ones,
and every flag (`--models`, `--projection tsne`, `--use-case-text`, `--list-models`, …).

## Conventions carried over from academic_research_agent

- **NULL is not 0.** "We never found out" and "we looked and there was nothing" are
  different facts — keep that distinction through cleaning and feature engineering
  instead of silently filling nulls with zero.
- **Cross-validated, not train-then-score.** Any accuracy/AUC number reported anywhere in
  this repo should be on held-out folds, never on the data the model was fit on.
- **Label which critic is speaking.** When a deterministic metric (e.g. CV ROC-AUC) and a
  qualitative judgement (e.g. "this looks like a good split") sit side by side, say which
  is which — a reader can't otherwise tell a measurement from an opinion.
