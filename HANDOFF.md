# TIRI — session handoff

Context primer for a new session. This file orients a fresh window — it doesn't itself
need to be kept up to date once superseded, but it was rewritten as of this handoff to
reflect the branch below rather than link-rot into confusion.

## What TIRI is

Data-science working repo built on top of the sibling repo `academic_research_agent`
(**never modify that repo from TIRI sessions** — TIRI only reads its exports as input
data; all work happens inside `TIRI---The-Data-science-project/`). The agent produces
labelled, ML-ready parquet exports (title/abstract/metadata, `triage_label`/
`review_label`, a precomputed `embedding` + `embed_model`); TIRI turns those into a
month-long ML workflow (README.md has the full 4-week plan: clean → feature-engineer →
model → ensemble → evaluate).

## Current work (on `main`, uncommitted): evaluation-metrics rework

A methodology review of the three comparison scripts found `roc_auc` (fit on paper
vectors alone, never sees the use-case query) and the centroid percentile (sees the
query, never sees labels) left a real gap — neither one answers "does this
representation retrieve the right papers for THIS query." Fixed in
`embedding_utils.py`/`latent_space_utils.py`, re-run on both real corpora, full trail
(what changed, what the new numbers say, what old claims didn't survive) in
`reports/metrics_rework_and_rerun.md` — **read that file, not this summary, before
trusting any AUC number in this repo.** Headlines:

- Every comparison script now reports **two AUC families** (`roc_auc_*`, classifier on
  paper vectors — the old metric; `query_auc_*`, ranking by cosine-to-use-case — new)
  under **two label readings** (`_strict` = positive vs. negative only; `_conservative` =
  positive vs. negative+pass), plus `roc_auc_std` (fold-level spread, previously computed
  then discarded) and `recall_at_Xpct`/`wss_at_95` (standard citation-screening metrics).
  `centroid_analysis` now also splits by class (`use_case_to_positive_centroid_sim` /
  `_negative_centroid_sim` / `_discriminative_gap`), not just the corpus-wide pool.
- **On the soil corpus, every embedding model's `query_auc` is BELOW 0.5** (0.29–0.43) —
  ranking by similarity to the use case's own text is worse than random at separating
  accepted from rejected papers, for all four models, despite `roc_auc` calling them
  solid (0.77–0.80). NER's `entity_type_counts` — the representation the old `roc_auc`
  reading called the weakest thing tested — has the only positive `query_auc` (0.579) of
  any single representation on that corpus.
- A **paired significance test** (new: `notebooks/comparisons/run_comparisons.ipynb` §4,
  using fold-level AUCs that used to be computed then thrown away) found most of this
  repo's old "X beats Y" point-difference claims are not statistically distinguishable
  from fold-to-fold noise on 5 folds — including `combined_features_notes.md`'s "combining
  made things worse" claim on the climate corpus (p=0.475–0.673).
- **`notebooks/comparisons/run_comparisons.ipynb` is now the primary way to run and see
  this repo's comparisons** — clone, open, run all cells, every table/plot renders
  inline. See `notebooks/README.md`.
- **`reports/` was cleaned out down to its `.md` decision-trail files** (`model_shortlist.md`,
  `ner_model_notes.md`, `combined_features_notes.md`, `metrics_rework_and_rerun.md`) —
  every per-run `_comparison.csv`/`_latent_space_comparison.png` (base and
  `richer_usecase` variants) was deleted. The three `compare_*.py` scripts no longer write
  a file by default — `--out`/`--out-plot` must be passed explicitly if a CSV/PNG is
  wanted for some other purpose. The `.md` files' own numeric tables are unaffected (the
  numbers are copied into the prose, not read from those files) — only their "run it
  yourself"/"outputs" pointer sentences were updated to point at the notebook.
  `future_work/train_baseline_classifier.py`'s `reports/*_baseline_*` files are untouched
  (out of scope — that script is unrelated to the comparison notebook).
- **`data/raw/sample-export.parquet` (the "climate/agriculture" 100-paper corpus) is
  removed from the repo.** It was a placeholder used to start testing before this
  project had real use-case briefs — no real use-case text of its own, too few
  observations to trust a comparison drawn from it, and it shouldn't be used for future
  model comparisons. Every `.md` report's climate-corpus numbers are kept as historical
  record with a notice marking them non-reproducible now. In its place, the notebook
  gained a second corpus source: `data/processed/papers_combined.parquet` — 2,873 papers
  across 6 real research questions (`cement_binders`, `soil_microbiome`, `ner`,
  `solar_leo`, `carbon_capture`, `tech_forecasting`, see `data/processed/README.md`) in
  one file, each carrying its own real objective + must-include/nice-to-have search terms.
  `notebooks/comparisons/run_comparisons.ipynb`'s `CORPUS_KEY` now takes `"soil"`
  (unchanged) or `"combined"` (pick a `USE_CASE_KEY`); for `"combined"`, the notebook
  builds a richer use-case query straight from those objective/terms columns
  (`build_richer_use_case_text`) instead of a short placeholder name — no separate
  `.usecase.json` file needed the way the standalone soil export required one.

## Prior branch `EMsanitycaheck` (merged into `main` via PR #18) — decoupling comparison from baseline selection

Prior sessions had built `scripts/compare_embeddings.py` (a latent-space/cosine-
similarity comparison of embedding models against a labelled export) and then treated its
"winner" as the Week-3 baseline classifier's default model
(`scripts/train_baseline_classifier.py`). This session's explicit goal: **that coupling
was wrong** — the comparison scripts exist to test how a use-case representation relates
to an already-labelled dataset (latent-space sanity, cosine similarity, use-case
centrality) for its own sake, not to pre-select a model for future classifier work. This
branch:

1. **Retired the "winner → baseline" framing everywhere it appeared** — script
   docstrings, `README.md`, `reports/model_shortlist.md` §5. The comparison scripts now
   say explicitly, in their own docstrings, that they're diagnostics, not a
   model-selection pipeline.
2. **Moved `train_baseline_classifier.py` to `future_work/train_baseline_classifier.py`**
   — kept functional (fixed its import of `embedding_utils.py`, which stayed in
   `scripts/`), but explicitly parked: not wired to either comparison script, its default
   model is a placeholder not a decision, and it's out of the current workflow.
3. **Extracted the latent-space diagnostic/plotting code out of `compare_embeddings.py`**
   into `scripts/latent_space_utils.py` (dispersion metrics, centroid analysis, 2D
   projection, the 3-panel plot) — representation-agnostic, so a second comparison script
   could reuse it instead of duplicating ~150 lines.
4. **Built `scripts/compare_ner_models.py`** — the same latent-space/cosine-similarity
   diagnostic, applied to NER-derived representations (spaCy `en_core_web_sm`) instead of
   sentence embeddings: an entity-type-count profile, and a TF-IDF-over-entity-text
   representation. Tested on both existing real corpora — full trail in
   `reports/ner_model_notes.md`. Headline findings worth carrying forward:
   - The use case's own short NAME text produced **zero entities** on the
     climate/agriculture corpus (a generic phrase has no `PERSON`/`ORG`/`GPE` for general
     NER to find) — collapsing `use_case_to_centroid_sim` to exactly 0.0 for both
     representations there. It was NOT zero on the soil-microbiome corpus's use-case name.
     Don't assume either outcome transfers to a new export.
   - 20-29% of papers (across both corpora) produced zero entities at all — a
     spaCy-on-this-text weakness, not a bug, and not representation-specific.
   - ROC-AUC for both NER representations stayed close to chance (0.55-0.62) on both
     corpora — meaningfully weaker than every embedding model tested in
     `model_shortlist.md` §4 (0.65-0.80). As a diagnostic reading (not a
     baseline-selection signal), generic NER-derived features alone don't separate these
     labels as well as sentence embeddings do, standalone. Untested: whether NER features
     help when concatenated ONTO an embedding rather than used alone.
5. Added `spacy>=3.7.0` to `requirements.txt` (`en_core_web_sm` needs a one-time
   `python -m spacy download en_core_web_sm` after `pip install`).
6. **Got the real use-case file and tested "richer use-case text" for real.**
   `data/raw/high-quality-microbial-and-fungal-community-in-soil.usecase.json` — the
   analyst's actual objective/key-terms JSON, tracked via a narrow `.gitignore` exception
   (`!data/raw/*.usecase.json`, alongside the existing `.gitkeep` exceptions) — is now in
   the repo. Built a fuller use-case string from its `objective` + `terms.must_include` +
   `terms.nice_to_have` fields and re-ran both comparison scripts on the soil corpus with
   `--use-case-text`, outputs at `reports/high-quality-microbial-and-fungal-community-in-soil-labelledFULLRUN_richer_usecase_*`.
   Full trail in `model_shortlist.md` §4b and `ner_model_notes.md` §4. Headline: **there's
   no single "richer text helps" verdict.**
   - ROC-AUC and dispersion are mathematically invariant to `--use-case-text` (confirmed
     identical before/after) — only the centrality panel can move; don't expect the other
     numbers to.
   - For embeddings, richer text moved centrality upward for every model tested, but by
     wildly different amounts — Specter jumped 29th→98.5th percentile, `bge-small` barely
     moved (5th→10.8th, stayed an outlier).
   - For NER, richer text moved the two representations in OPPOSITE directions:
     `entity_type_counts` got LESS central (70th→39th), `entity_text_tfidf` got MORE
     central (89th→100th). Richer text added domain words (helps entity_text_tfidf) but
     not more recognisable spaCy entity TYPES (dilutes entity_type_counts).
   - Practical implication: use-case wording is a real, uncontrolled variable that can
     swing centrality by up to 70 percentile points — any future use of centroid
     similarity as a feature (not just a diagnostic) needs that wording fixed and
     documented, not left to whichever column happened to be read that day.

## What's been done before this branch (embedding model selection)

1. **Priority properties table** (`reports/model_shortlist.md` §1) — 11 properties that
   actually matter for this workflow (symmetric vs. asymmetric semantics, latent-space
   sanity, multilinguality, discriminative power on real labels, local/offline
   constraint, throughput, license, etc.), derived from the agent's own code/docs, not a
   generic embedding-benchmark checklist.
2. **Ranked 10-model candidate list** (§2) spanning domain-specific (Specter), corrected
   asymmetric (bge, e5), general MTEB leaders (gte, mxbai), multilingual alternatives
   (LaBSE), a tiny/fast floor (all-MiniLM-L6-v2), and an API ceiling reference (OpenAI,
   explicitly disqualified — breaks the local/offline/no-API-key constraint).
3. **Built `scripts/compare_embeddings.py`** — visually compares the *latent space* of
   several models on a labelled export: a 2D PCA/t-SNE map, a pairwise-cosine dispersion
   histogram (dense/collapsed vs. disperse — anisotropy), and a use-case-to-centroid
   histogram (how typical is the use case's own phrasing vs. the corpus it found).
   Secondary metric: cross-validated ROC-AUC (same method as the agent's own
   `model.py:_cross_validated_roc_auc`) — a diagnostic on the labelled dataset, not a
   model-selection signal (see this branch's changes above).
4. **Tested on 2 real corpora** (climate/agriculture, 100 papers; soil microbiome, 602
   papers) — full results and non-obvious findings in `reports/model_shortlist.md` §3-4.
   Headlines worth carrying forward:
   - The model ranking **flipped** between the two corpora — the agent's current default
     was best on the small corpus, worst (though still solid) on the larger one. Model
     choice looks corpus-dependent; don't generalize from one dataset.
   - **Dispersion and discriminative power don't track together** — `bge-small-en-v1.5`
     looked the most collapsed by the dispersion metric on both corpora, yet had the best
     ROC-AUC on the larger one. Don't rule a model out on dispersion alone.
   - **Specter** (the domain-specific, citation-trained model) scored at chance level on
     the smaller corpus even after fixing its documented `[SEP]`-join input-format quirk —
     a real negative result, not an artifact.
   - `bge-small`'s use-case-to-centroid percentile was an extreme outlier on *both*
     corpora — flagged as possibly an artifact of comparing an asymmetric model's prefixed
     query against unprefixed passages, not necessarily a real topical mismatch.
   - `nomic-embed-text-v1.5` was tested and dropped from the default set (~795s to embed
     on CPU vs. 4-85s for everything else) — still registered, just not run by default.
5. **A prior session picked `paraphrase-multilingual-MiniLM-L12-v2` and built a Week-3
   baseline on it** (`reports/model_shortlist.md` §5) — that pick/framing is what this
   branch retired (see above); the plumbing survives, parked, in
   `future_work/train_baseline_classifier.py`.

## Current repo state

```
scripts/
  embedding_utils.py               shared: model registry+prefixes, cross-validation,
                                    resolve_paper_vectors (reuse export embeddings vs. re-embed)
  latent_space_utils.py            shared: dispersion/centroid diagnostics + plotting,
                                    used by ALL THREE comparison scripts below
  compare_embeddings.py            embedding-vs-labelled-dataset comparison (diagnostic only)
  compare_ner_models.py            NER-vs-labelled-dataset comparison (diagnostic only)
  compare_combined_features.py     embedding+NER concatenated, same diagnostics (diagnostic only)
future_work/
  train_baseline_classifier.py     parked Week-3 baseline plumbing — NOT wired to any
                                    comparison script, default model is a placeholder
notebooks/
  comparisons/run_comparisons.ipynb    same comparison functions, output inline — see notebooks/README.md
reports/
  metrics_rework_and_rerun.md      evaluation-metrics rework decision trail — READ THIS FIRST for AUC numbers
  model_shortlist.md               embedding decision trail (§1-5, §4b)
  ner_model_notes.md               NER decision trail — same structure, for NER
  combined_features_notes.md       does concatenating NER onto an embedding help? — mixed/no (roc_auc reading; see metrics_rework_and_rerun.md for the query_auc reading)
  <dataset>_baseline_confusion_matrix.png / _baseline_metrics.json / _baseline_scored_pool.csv
                                    (from future_work/train_baseline_classifier.py — unrelated
                                    to the 3 comparison scripts above, which write nothing by
                                    default now; run notebooks/comparisons/run_comparisons.ipynb
                                    to see their output instead of looking for a file here)
data/raw/
  high-quality-microbial-and-fungal-community-in-soil-labelledFULLRUN.parquet  (602 papers)
  high-quality-microbial-and-fungal-community-in-soil.usecase.json  (the analyst's real
                                                  objective/key-terms JSON for that corpus —
                                                  tracked via a narrow .gitignore exception)
                                                  (sample-export.parquet removed — see above)
data/processed/
  papers_combined.parquet          6 real research questions in one file, see README.md
                                    in that folder + notebooks/comparisons/run_comparisons.ipynb
```

The 3 comparison scripts write nothing to disk by default — run
`notebooks/comparisons/run_comparisons.ipynb` to see output, or pass `--out`/`--out-plot`
explicitly for a one-off file (name it after the input, `reports/<stem>_*`, if you do —
never a fixed path, so re-running on a different export doesn't overwrite prior results).

## Established conventions worth keeping

- **TIRI-only changes.** Read `academic_research_agent` for reference; never edit it.
- **`reports/` holds decision-trail `.md` files, not per-run artifacts.** The comparison
  scripts default to writing nothing; `notebooks/comparisons/run_comparisons.ipynb` is how
  you see a run's output. If you DO pass `--out`/`--out-plot` for a one-off file, name it
  after the input (`reports/<stem>_*`) — never a fixed path — but don't check that file in.
- **Cross-validated, never train-then-score.** Any AUC/accuracy number must come from
  held-out folds.
- **Report `roc_auc` alongside `query_auc`, never `roc_auc` alone** — the classifier
  version never sees the use-case query, the ranking version never needs training; they
  can and do disagree (see `reports/metrics_rework_and_rerun.md`'s soil-corpus result).
  Same for `_strict` vs. `_conservative`: don't drop `pass` rows without also reporting
  the reading that keeps them.
- **A fold-count difference is a claim, not a given — check `roc_auc_std` or run the
  notebook's paired test before calling one number "better" than another.**
- **NULL is not 0** — "never found out" vs. "looked, found nothing" are different facts.
- **Shared logic lives in a `*_utils.py` module**, not duplicated per script —
  `embedding_utils.py` (embedding-specific) and `latent_space_utils.py`
  (representation-agnostic diagnostics/plotting) both exist for this reason.
- **Comparison/diagnostic scripts don't select models for future modelling work.** State
  that scope explicitly in any new comparison script's docstring — this branch exists
  because that boundary got blurred once already.
- Report honest negative/surprising results (Specter's chance-level score, the ranking
  flip, NER's zero-entity use-case text) rather than only the flattering ones — this
  repo's decision trail is written that way on purpose so future-you can trust it.

## Where this is heading next

- **NER features concatenated onto an embedding — tested, result: no free win.**
  `scripts/compare_combined_features.py` (unit-normalises each block before `np.hstack`)
  found combining was neutral on the soil corpus (ROC-AUC 0.760/0.762 vs. 0.768 alone,
  noise-level) and actively worse on the climate corpus (0.605/0.629 vs. 0.650 alone) —
  full trail in `reports/combined_features_notes.md`. It DID reliably rescue the
  use-case centrality diagnostic from a degenerate all-zero-NER-block case, a real if
  narrower benefit than hoped for. Two things this run did NOT test, still open: (1) a
  tuned, non-50/50 block weighting (`--embedding-weight`, not yet built); (2) whether a
  trained *cross-encoder-style re-ranking* stage (the technique that actually drove the
  field-guide's +20% hit@5 example) behaves differently from static concatenation —
  those are two different techniques, don't conflate a negative result on one with the
  other.
- **"Use case transformer" — tested with the real file, result: no single verdict.**
  The export's short use-case NAME vs. the analyst's real objective/key-terms JSON
  (`data/raw/high-quality-microbial-and-fungal-community-in-soil.usecase.json`, now in
  the repo) was tested for real this branch — full trail in `model_shortlist.md` §4b and
  `ner_model_notes.md` §4. Richer text moved use-case centrality by up to 70 percentile
  points, but the direction and size depended entirely on which representation was
  asking (Specter: hugely more central; bge-small: barely moved; NER's two
  representations moved in OPPOSITE directions from each other). Still open: an
  LLM-expanded query or a learned transformation of the use case, neither attempted here
  — this branch only tested a hand-built string from an existing JSON file, not a new
  transformation technique.
- **Domain-trained NER as a follow-up to the spaCy floor test.** `compare_ner_models.py`
  used `en_core_web_sm` (generic entity types, no PyTorch dependency) as a deliberate
  floor. `reports/ner_model_notes.md` §2 flags scispaCy (biomedical-trained) and GLiNER
  (zero-shot, custom domain labels like "organism"/"technique") as the natural next
  candidates if this floor's ROC-AUC (0.55-0.62, weaker than every embedding tested) and
  zero-entity rate (20-29%) turn out to matter for downstream feature engineering — still
  untested, unchanged from before this branch's combined-features work.
- **Week 3 baseline-classifier work itself** — genuinely unstarted. Which features
  (embedding alone? NER-augmented? metadata?), which model, how to validate it, is a
  decision for whenever that week's work actually begins — deliberately NOT pre-decided
  by either comparison script per this branch's whole point.
