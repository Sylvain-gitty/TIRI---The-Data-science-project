# Embedding model shortlist — properties, ranking, and what we're actually testing

> **Status: current** as the decision trail for `compare_embeddings.py`'s default model set — a *diagnostic* against the labelled corpus, not the production embedding choice. For the models actually shipped (Jasper + Qwen3-8B) see [`wf_ensemble_final_recommendations.md`](wf_ensemble_final_recommendations.md) #4. AUC readings here predate [`metrics_rework_and_rerun.md`](metrics_rework_and_rerun.md).

This is the decision trail behind `scripts/compare_embeddings.py`'s default model set,
in three parts: the properties we're optimising for, the full ranked candidate list they
came from, and the models we selected to actually run — with the concrete config
(backend, prefixes) each one runs under, and what testing them actually found.

**Scope note:** this comparison is a diagnostic exercise against an already-labelled
dataset — latent-space sanity, cosine similarity, use-case centrality (see
`scripts/compare_embeddings.py`'s docstring) — not a pipeline for selecting a future
classifier's model. §5 below used to frame its pick as "the Week-3 baseline"; that framing
has been retired — nothing downstream in this repo is wired to whatever
"wins" a run of this script. See `reports/ner_model_notes.md` for the same kind of
exercise applied to NER-derived representations instead of embeddings.

> **Corpus retired (2026-08-04):** `data/raw/sample-export.parquet` — the
> "climate/agriculture" 100-paper corpus §3 and most of §5 below are about — has been
> removed from this repo. It was a placeholder used to start testing before this project
> had real use-case briefs: no real use-case text of its own (just a generic short name)
> and too few observations to trust a comparison drawn from it. The numbers below are kept
> as historical record and can no longer be reproduced by re-running against that file.
> New comparisons should use the soil-microbiome corpus (§4) or
> `data/processed/papers_combined.parquet`'s 6 real research questions — see
> `notebooks/experiments/run_comparisons.ipynb`.

## 1. Priority properties (why these, for THIS workflow)

Not a generic embedding-benchmark checklist — grounded in `academic_research_agent`'s own
constraints (`embeddings.py`, `model.py`, `CLAUDE.md`).

| # | Property | Why it matters for the academic-agent workflow | Priority |
|---|---|---|---|
| 1 | **Symmetric vs. asymmetric semantics** | The agent embeds the use case and every paper abstract the same way and compares by cosine similarity. Asymmetric models (bge/e5-family) expect a `"query: "` / `"passage: "` prefix — skip it and the vectors collapse into a narrow cone. | Must-have / must-configure-correctly |
| 2 | **Latent-space sanity (dense vs. disperse)** | Cosine ranking is the whole mechanism — triage order, "similar to kept papers," the classifier fallback. A collapsed space makes every paper look equally (ir)relevant. | Must-have |
| 3 | **Multilinguality** | Explicit design goal: a scout shouldn't miss a good paper for being written in German or Chinese. | Must-have (given the product's stated goal) |
| 4 | **Discriminative power on YOUR labels** | The end use is ranking/classifying papers by *your* triage decisions, not a generic MTEB score. | Must-have |
| 5 | **How central the use case sits vs. its own corpus** | If the use case embedding is an outlier relative to the papers found for it, that's a signal the query vocabulary doesn't match the literature's. | Should-have |
| 6 | **Local, offline, no API key** | Hard architectural constraint — no external service or API key, entirely offline. | Must-have (hard constraint) |
| 7 | **CPU throughput** | Retrieval embeds a whole pool before trimming ("embed-then-trim") — has to stay interactive, no GPU assumed. | Should-have |
| 8 | **Model download size / disk footprint** | Distributed to pilot users as a git clone — a multi-GB model is a real onboarding cost. | Should-have |
| 9 | **Embedding dimensionality** | Vectors are stored in a `pgvector` column keyed by model; dimension is a schema/storage decision, not a free parameter. | Should-have |
| 10 | **License compatibility** | The project has already made one licensing call under this exact pressure (PyMuPDF/AGPL → pypdf). | Should-have |
| 11 | **Version/backend stability** | Vectors are tagged `model@fastembed-version` because a library upgrade can silently change pooling and invalidate old vectors. | Nice-to-know |

## 2. Full ranked candidate list (10 models considered)

| Rank | Model | Source | Dim | Symmetric / prefix needed? | Why this rank |
|---|---|---|---|---|---|
| 1 | `allenai/specter2` (via `sentence-transformers/allenai-specter`) | sentence-transformers | 768 | Symmetric | Domain match: trained on paper citation pairs — closest thing to this exact task on the list. |
| 2 | `BAAI/bge-small-en-v1.5` (re-run **with** its query prefix) | fastembed | 384 | Asymmetric — the fix | Retest of a bug, not a new model: isolates whether the missing prefix caused last round's collapse. |
| 3 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (current default) | fastembed | 384 | Symmetric | The control every other row is compared against. |
| 4 | `thenlper/gte-base` | fastembed | 768 | Symmetric, no prefix | Strong general MTEB performer with no prefix discipline required. |
| 5 | `intfloat/multilingual-e5-large` | fastembed | 1024 | Asymmetric (`query:`/`passage:`) | Multilingual challenger, done correctly (prefixed) from the start. |
| 6 | `nomic-ai/nomic-embed-text-v1.5` | fastembed | 768 (Matryoshka, truncatable) | Task-prefixed | Only model with native variable dimensionality — tests the storage/dim tradeoff directly. |
| 7 | `mixedbread-ai/mxbai-embed-large-v1` | fastembed | 1024 | Symmetric-ish | Top-tier open English embedding — tests whether bigger/newer buys real separation. |
| 8 | `sentence-transformers/LaBSE` | sentence-transformers | 768 | Symmetric | Multilingual via a different training philosophy (translation pairs vs. paraphrase pairs). |
| 9 | `sentence-transformers/all-MiniLM-L6-v2` | fastembed | 384 | Symmetric | The tiny/fast floor — is the extra size elsewhere earning its keep? |
| 10 | `OpenAI text-embedding-3-small` | OpenAI API | 1536 (truncatable) | Either | Reference-only ceiling — **disqualified** for actual use (breaks the local/offline/no-API-key constraint). |

## 3. Selected for this round: models 1, 2, 3, 9 (6 tested once, then dropped)

The subset actually wired into `scripts/compare_embeddings.py`'s `DEFAULT_MODELS`, in the
order they run (matches the ranking above).

| Rank | Model | Backend | Dim | `query_prefix` | `passage_prefix` | What this run is meant to answer |
|---|---|---|---|---|---|---|
| 1 | `sentence-transformers/allenai-specter` | sentence-transformers | 768 | *(none — symmetric)* | *(none — symmetric)* | Does a domain-specific, citation-trained model beat general-purpose ones on discriminative power (ROC-AUC) and latent-space sanity? |
| 2 | `BAAI/bge-small-en-v1.5` | fastembed | 384 | `"Represent this sentence for searching relevant passages: "` | *(none)* | Was last round's 0.77-avg-cosine "collapse" a missing-prefix artifact? This is the corrected retest. |
| 3 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | fastembed | 384 | *(none — symmetric)* | *(none — symmetric)* | Control: the agent's current default, unchanged, for every other row to be compared against. |
| 9 | `sentence-transformers/all-MiniLM-L6-v2` | fastembed | 384 | *(none — symmetric)* | *(none — symmetric)* | The tiny/fast floor — is anything above actually earning its extra size? |

**#6, `nomic-ai/nomic-embed-text-v1.5`, was tested and then removed from the default
set** (still registered in `MODEL_CONFIGS`, runnable via `--models` if wanted): on a
100-paper test corpus it took ~795s to embed vs. 4-85s for every other model above —
fastembed's ONNX graph for this model is dramatically slower than the rest on CPU, which
outweighed what its Matryoshka dimensionality-tradeoff answer was worth for a script
meant to be re-run often while iterating.

**What retesting #1 and #2 actually found (worth knowing before trusting a re-run):**
- **#2 (bge-small, corrected prefix):** bge's own convention prefixes ONLY the query,
  never the passages — so the fix left the corpus-wide dispersion completely unchanged
  (0.7692 both times). The "prefix caused the collapse" hypothesis is NOT confirmed for
  the corpus itself; bge-small's document embeddings really do look collapsed on the
  climate/agriculture test corpus, prefix or no prefix.
- **#1 (Specter):** its model card documents joining title+abstract with `[SEP]`, not
  `". "` — fixed via `title_abstract_sep`. Barely moved either number (avg cosine
  0.80 -> 0.81, ROC-AUC 0.4947 -> 0.4996): still the most collapsed, still chance-level.
  The domain-specific model did not outperform on this corpus even once its known input
  quirk was corrected — a real, if unflattering, result.

**Run it yourself, with output inline:** open `notebooks/experiments/run_comparisons.ipynb`
(from inside `notebooks/experiments/`), set `CORPUS_KEY`, run all cells — every table and
plot renders in the notebook itself, nothing is written to `reports/` by default. See
`notebooks/README.md`.

`scripts/compare_embeddings.py` is the library these functions live in (the notebook
imports it, doesn't reimplement it) and still works as a CLI for scripting/automation —
`python scripts/compare_embeddings.py --data data/raw/your-export.parquet` prints the same
metrics table to the console; pass `--out`/`--out-plot` explicitly if you also want a CSV
or PNG written to disk (neither is written unless asked, on purpose — `reports/` holds
this decision trail, not a growing pile of per-run artifacts).

## 4. Second corpus tested (602-paper soil microbiome export) — the ranking flipped

Re-running the same 4 shortlisted models on a much larger, different-domain export
(`high-quality-microbial-and-fungal-community-in-soil-labelledFULLRUN.parquet`, 602
papers, 94 positive / 263 negative) than the climate/agriculture one (100 papers):

| Model | avg cosine | ROC-AUC | use-case percentile | embed time |
|---|---|---|---|---|
| `allenai-specter` | 0.84 — collapsed | 0.776 | 29th | 226s |
| `bge-small-en-v1.5` | 0.78 — collapsed | **0.803 — best here** | 5th — outlier | 200s |
| `paraphrase-multilingual` (control) | 0.56 — moderate | 0.768 — **worst here** | 81st — very central | 38s |
| `all-MiniLM-L6-v2` | 0.48 — best dispersion | 0.793 | 56th | **18s — fastest** |

The control had the BEST ROC-AUC on the small climate corpus (0.650) and the WORST on
this larger soil corpus (0.768, though still a solid score) — the ranking flipped between
the two real datasets tested. Two non-obvious things worth carrying forward:
- Dispersion and discriminative power don't necessarily track together: `bge-small`
  looks the most collapsed by the dispersion metric on both corpora, yet has the best
  ROC-AUC on the larger one. Don't rule a model out on dispersion alone.
- `bge-small`'s use-case-centroid percentile was an extreme outlier on BOTH corpora
  (7th, then 5th) — worth treating as a possible artifact of comparing an asymmetric
  model's query embedding against unprefixed passage embeddings (the centroid-vs-use-case
  metric assumes a shared geometry that symmetric models guarantee and asymmetric ones
  don't), not necessarily a real topical mismatch.

## 4b. The use-case TEXT matters as much as the model — tested with the real use-case file

Every result above used the export's short `use_case` NAME column
("High quality microbial and fungal community in Soil"). The actual objective/key-terms
JSON behind that corpus
(`data/raw/high-quality-microbial-and-fungal-community-in-soil.usecase.json` — the
analyst's real use-case definition, not the export's abbreviated name) was used to build
a fuller `--use-case-text` string (`objective` + `terms.must_include` +
`terms.nice_to_have`, space-joined) and re-run against the same 602-paper corpus, same 4
models, outputs at `reports/high-quality-microbial-and-fungal-community-in-soil-labelledFULLRUN_richer_usecase_*`.

| Model | use-case percentile (short name) | use-case percentile (full objective+terms) | ROC-AUC / dispersion |
|---|---|---|---|
| `allenai-specter` | 29th (sim 0.68) | **98.5th (sim 0.96)** — huge jump | unchanged: 0.776 / 0.84 (collapsed) |
| `bge-small-en-v1.5` | 5th (sim 0.85) | 10.8th (sim 0.84) — still an outlier | unchanged: 0.803 / 0.78 (collapsed) |
| `paraphrase-multilingual` (control) | 81st (sim 0.85) | 88.5th (sim 0.82) | unchanged: 0.768 / 0.56 (moderate) |
| `all-MiniLM-L6-v2` | 56th (sim ~0.73) | 63.8th (sim 0.73) | unchanged: 0.793 / 0.48 (best dispersion) |

Two things worth carrying forward:

- **ROC-AUC and dispersion are mathematically invariant to `--use-case-text`** — confirmed
  identical to rounding, before vs. after, for every model. This isn't a coincidence to
  re-verify each time: `cross_validated_roc_auc`/`dispersion_metrics` only ever see paper
  vectors + labels, the use-case vector never enters that calculation. Only the
  centrality panel (`use_case_to_centroid_sim` / percentile) can move when
  `--use-case-text` changes — don't expect the other numbers to.
- **Richer text moved centrality a lot, but not uniformly, and mostly upward.** Specter's
  jump (29th→98.5th) is the standout — going from "looks like an outlier relative to its
  own corpus" to "looks like one of the most prototypical papers in it," just from
  swapping a 6-word name for the real objective + key terms. `bge-small` barely moved and
  stayed an outlier both times, consistent with §4's standing hypothesis that its outlier
  behaviour is a geometry artifact (asymmetric prefix handling) rather than a genuine
  topical signal that more text would fix. There is no single "richer text helps"
  verdict — the size of the effect is model-specific.

This matters beyond curiosity: if a future step ever uses use-case-centroid similarity as
a *feature* (not just a diagnostic), the wording fed into `--use-case-text` is a real,
uncontrolled variable that can swing results by up to 70 percentile points — it needs to
be fixed and documented, not left to whichever column happened to be read that day.

## 5. Which model looked most consistent across the 2 corpora tested (observation, not a pick)

**`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** (the control /
academic_research_agent's current default) held up most consistently across the two real
corpora tested, weighing the evidence above against the priorities in §1:

1. **Most consistent across both real corpora tested.** Never chance-level (Specter was,
   once), never the most collapsed (bge was, both times) — solid and unsurprising both
   times (0.650, then 0.768), even where it wasn't the single best score.
2. **Symmetric — no prefix-config fragility.** bge and nomic need their prefix scheme
   maintained correctly to mean anything (see §3's retest); a model that doesn't need
   that is one less thing to get subtly wrong later.
3. **Matches the actual (if small) multilingual fraction in both real exports tested** —
   2-3 non-English papers out of ~100-600 each time. Not large, but real, and priority #3
   is an explicit product goal, not a hypothetical.
4. **Free reuse of the export's own precomputed vectors.** Both real exports' `embed_model`
   column is exactly this model — `embedding_utils.resolve_paper_vectors` reuses those
   vectors directly instead of re-embedding, which is both faster AND uses the literal
   vectors academic_research_agent's own UI already ranked by, not a fresh approximation
   of them.

This is a judgement call from 2 data points about the DIAGNOSTIC itself, not a model pick
for future modelling work — `--model`/`--models` overrides it any time a different
corpus's own comparison run (§4's table) suggests otherwise. Week 3's actual
baseline-classifier work (which features, which model, how to validate it) is unstarted,
unscoped, and intentionally NOT decided by this exercise — see `future_work/
train_baseline_classifier.py` for plumbing that used to be wired to this section's old
"winner" framing and is now parked, decoupled, and unused by the current workflow.
