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
`scripts/compare_embeddings.py` and `scripts/compare_ner_models.py` below — this is the
week to run them and inspect the latent space of each candidate embedding/NER
representation against the already-labelled dataset: is the space sane or collapsed, how
central does the use case sit relative to its own corpus, does it separate the existing
labels at all). This is a diagnostic exercise, not a model-selection step for Week 3 —
see the scope note in each script's docstring. Metadata features: citation counts,
venue/source one-hots, recency.

**Week 3 — Modelling**
Baselines and ensembles (random forest, gradient boosting, stacking) — unstarted,
unscoped as of this writing. Which features/model the baseline trains on is a decision
for that week, not something Week 2's embedding/NER comparison pre-selects. Cross-validate
honestly (`StratifiedKFold`, no leakage between folds) — a score on data the model was
trained on is not a result. `future_work/train_baseline_classifier.py` has cross-
validation/scoring plumbing kept from an earlier, now-retired framing where it was wired
to Week 2's comparison — functional but parked, not part of the current workflow.

**Week 4 — Evaluate, error-analyse, write up**
Held-out evaluation, error analysis (which papers does the model get wrong, and why),
compare against the agent's own relevance ranking as a baseline, final report.

## Repo layout

```
data/
  raw/          exports dropped in as-is (gitignored — data doesn't belong in git)
  processed/    cleaned / feature-engineered outputs of your own pipeline
notebooks/      see notebooks/README.md for the current list and what each one does
  eda/            exploratory analysis — read-only, no files written back to data/
  data_compile/   data-processing pipelines — data/raw/ -> data/processed/
scripts/        standalone, runnable analysis scripts
  embedding_utils.py       shared embedding logic (model registry, prefixes, cross-
                            validation) behind compare_embeddings.py — not run directly
  latent_space_utils.py    shared latent-space diagnostics + plotting (dispersion,
                            centroid analysis, 2D projection) behind BOTH comparison
                            scripts below — not run directly
  compare_embeddings.py    week-2 embedding-vs-labelled-dataset comparison (see below)
  compare_ner_models.py    week-2 NER-vs-labelled-dataset comparison (see below)
future_work/    parked, deferred work — not part of the current workflow
  train_baseline_classifier.py   Week-3 baseline plumbing, kept but unwired (see below)
reports/        write-ups, figures, model comparison tables
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Scripts

Both comparison scripts below share one scope note, worth stating once: they test how a
representation (an embedding model, or an NER-derived feature) relates to an
**already-labelled dataset** — latent-space sanity, cosine similarity, use-case
centrality — purely as a diagnostic. Neither one selects a model for a future classifier
baseline; nothing downstream in this repo is wired to whatever "wins" a run of either
script. See `future_work/train_baseline_classifier.py` below for where that used to not
be true, and why it's parked now.

### `scripts/compare_embeddings.py`

Visually compares the **latent space** of several embedding/NLP models against your
labelled export — a lexical TF-IDF baseline plus one or more fastembed neural models by
default. For each model it draws a row of three plots:

1. a 2D map of the whole corpus (PCA or t-SNE), coloured by triage label, with the
   corpus centroid and the embedded use case marked;
2. a pairwise-cosine-similarity histogram — is this model's latent space **dense**
   (every paper's vector points the same way — a known failure mode, "anisotropy"/
   representation collapse) or **disperse** (vectors actually spread out by content)?
3. a histogram of how similar each abstract is to the corpus centroid, with the use
   case's own similarity marked — **how central is the use case relative to its own
   corpus?**

It also reports cross-validated ROC-AUC (same lens as the agent's own adaptive triage
classifier, `model.py`) as a secondary "does this space separate the labels you already
have" diagnostic — a reading on the labelled dataset, not a signal for picking a future
classifier's model (see scope note above).

```bash
python scripts/compare_embeddings.py --data data/raw/your-export.parquet
```

Outputs (named after the input file, so different exports never overwrite each other):
`reports/<data filename>_latent_space_comparison.png` (the visual comparison) and
`reports/<data filename>_embedding_comparison.csv` (the scalar metrics behind it). See the
script's own docstring for the full walkthrough — what each metric means, why those
specific ones, and every flag (`--models`, `--projection tsne`, `--use-case-text`,
`--list-models`, `--out`/`--out-plot`, …). See also `reports/model_shortlist.md` for the
reasoning behind the default model set and what testing each one actually found.

### `scripts/compare_ner_models.py`

The same latent-space/cosine-similarity comparison as `compare_embeddings.py`, but for
**NER-derived representations** instead of sentence embeddings — extracting named
entities (organisations, locations, dates, ...) from titles/abstracts via a local spaCy
pipeline (`en_core_web_sm` by default) and testing whether turning a paper into an
entity-based vector produces a sane latent space, and where the use case lands in it,
against the same labelled dataset. Two representations by default: an entity-type-count
profile, and a TF-IDF-over-entity-text representation — same 3-panel-per-row plot and
scalar metrics as the embedding comparison, via shared `scripts/latent_space_utils.py`.

```bash
python scripts/compare_ner_models.py --data data/raw/your-export.parquet
```

Outputs: `reports/<data filename>_ner_latent_space_comparison.png` and
`reports/<data filename>_ner_representation_comparison.csv`. See the script's own
docstring for every flag (`--spacy-model`, `--representations`, `--list-entity-labels`,
…) and `reports/ner_model_notes.md` for why spaCy was picked over GLiNER/scispaCy and
what testing it on both real corpora actually found — including the honest negative
result that a short use-case NAME often yields zero recognisable entities.

Requires the spaCy model to be downloaded once after `pip install`:
```bash
python -m spacy download en_core_web_sm
```

### `future_work/train_baseline_classifier.py` — parked, not part of the current workflow

Originally built as "the Week-3 baseline," with its default model framed as the winner
of `compare_embeddings.py`'s comparison. That framing is retired (see `HANDOFF.md`): the
comparison scripts are diagnostics against the labelled dataset, not a model-selection
step for future classifier work, and Week 3's actual baseline (which features, which
model, how to validate it) hasn't been decided. The script still runs — a plain
`LogisticRegression` on paper embeddings, cross-validated honestly (out-of-fold
predictions, never a row graded by a model that trained on it), scoring every paper with
a usable vector — kept here so that plumbing isn't lost, but it is not wired to either
comparison script's output and its default model is a placeholder, not a decision.

```bash
python future_work/train_baseline_classifier.py --data data/raw/your-export.parquet
```

## Conventions carried over from academic_research_agent

- **NULL is not 0.** "We never found out" and "we looked and there was nothing" are
  different facts — keep that distinction through cleaning and feature engineering
  instead of silently filling nulls with zero.
- **Cross-validated, not train-then-score.** Any accuracy/AUC number reported anywhere in
  this repo should be on held-out folds, never on the data the model was fit on.
- **Label which critic is speaking.** When a deterministic metric (e.g. CV ROC-AUC) and a
  qualitative judgement (e.g. "this looks like a good split") sit side by side, say which
  is which — a reader can't otherwise tell a measurement from an opinion.

## Collaboration conventions

### Branch naming

Prefix every branch with your initials, then a short, descriptive slug:
`<initials>-<what-it-does>` — e.g. `wf-eda-notebooks`, `wf-data-compile`. In a
2-person repo where several branches can be in flight at once, this makes
`git branch -a` self-explanatory about who's working on what, without having to open
each branch to check.

### Documentation standards

- Every notebook gets a one-line description in `notebooks/README.md`, kept current
  with the actual folder structure (`eda/`, `data_compile/`, ...) — that file is the
  source of truth for "what notebooks exist and what do they do", not this README.
- Notebooks explain *why*, not just *what*, in markdown cells, and label which claims
  are hard computed numbers vs. judgement calls (same "label which critic is
  speaking" convention as above).
- Scripts keep the existing convention: a long module docstring carrying the
  reasoning — and any negative results from real runs — behind the choices made (see
  `scripts/compare_embeddings.py` for the reference example).
