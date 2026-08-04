# Notebooks

Organised by what a notebook does, not by week — a notebook's folder tells you
whether it's safe to just read (`eda/`), whether it writes to `data/processed/`
(`data_compile/`), or whether it's a small viability check for a candidate feature
that isn't part of the model yet (`feature_experiments/`).

```
notebooks/
  eda/                  exploratory analysis — read-only, no files written back to data/
  data_compile/         data-processing pipelines — data/raw/ -> data/processed/
  feature_experiments/  small, sample-sized viability checks for candidate features —
                        read-only, not the Week-3 modelling pipeline itself
```

## eda/

| Notebook | What it does |
|---|---|
| `eda_quickstart.ipynb` | **Start here.** Loads `data/processed/papers_combined.parquet` and runs a basic, deliberately simple first pass — shape, dtypes, missing values, `.describe()`, key value counts, and a handful of plain charts. A launching-off point for anyone new to the project, not a deep dive. |
| `explore_use_cases.ipynb` | Compares structure and volume across the 6 labelled JSONL exports in `data/raw/` — schema, nulls, label balance, duplicates, citation/abstract-length distributions. Read-only. |
| `explore_usecase_definitions.ipynb` | Compares the 6 `usecase.json` search-brief definitions (problem statement, objective, terms, TRL constraints, decision criteria) against their matching JSONL export — structure, and a table joining each definition against its resulting corpus. Read-only. |
| `wf_data_enrich.ipynb` | Descriptive EDA on `data/processed/papers_combined.parquet`: `notes` content, author/citation/venue breakdowns by use case and `triage_label`, an outlier review (author/citation extremes, long venue names), author frequency per use case, and an interactive citations-vs-age scatter (plotly) plus a citations-by-label-by-use-case boxplot. Read-only. |

## data_compile/

| Notebook | What it does |
|---|---|
| `combine_use_cases.ipynb` | Reads all 6 JSONL exports + their matching `usecase.json` search-brief definitions from `data/raw/`, applies the cleaning/standardisation punch list found by the `eda/` notebooks (compound `paper_id`, nullable `Int64` dtypes, `venue` casing, `sources` parsed into booleans, fixed `triage_label`/`review_label` categoricals, `abstract_source` dropped, `has_abstract` flag), broadcasts both files' metadata onto every paper row, and writes `data/processed/papers_combined.parquet` — the file `scripts/compare_embeddings.py` and `scripts/train_baseline_classifier.py` are meant to consume via `--data`. See also `data/processed/README.md` for a full description of the output dataset. |

## feature_experiments/

| Notebook | What it does |
|---|---|
| `author_orcid.ipynb` | Viability check for an "author prolificacy" feature: samples ~10 authors from `papers_combined.parquet` (known repeat authors + singleton-paper authors), resolves each to an ORCID iD via their paper's `doi` against the open Crossref API, and where found, reads a total-works count off the public ORCID record. Headline result is the ORCID coverage/match rate (20%, publisher-dependent not prominence-dependent), not the feature values themselves — not viable at that coverage level. Small, experimental, not a pipeline. |
| `terms_overlap.ipynb` | Viability check for a candidate feature: a per-paper term-overlap score against its own use case's `terms_must_include`/`terms_nice_to_have`/`terms_exclude` search-brief lists (whole-word matching). Tests whether the score separates `positive`/`negative` `triage_label`s and how it relates to `relevance_score` — real signal (ROC-AUC 0.70–0.80) on 3 of 6 use cases, none on the other 3. Read-only, no model trained. |
| `terms_overlap_spacy.ipynb` | Follow-up to `terms_overlap.ipynb`: does spaCy (lemma matching, phrase/hyphenation-normalized matching, negation-aware `terms_exclude`) beat plain whole-word regex matching? Best variant (hybrid surface-or-lemma) gives a small, safe +0.01 AUC gain on 2 of the 3 working use cases and rescues none of the 3 dead ones — marginal, not transformative; semantic/word-vector similarity scoped but not built (no headroom left to justify it, and this repo's own `combined_features_notes.md` already found generic spaCy-derived features add no reliable win alongside embeddings). Read-only, no model trained. |
| `trl_estimate.ipynb` | Viability check on a 34-paper hand-picked sample: can a paper's own Technology Readiness Level be estimated from its title + abstract, as a genuinely per-row feature (unlike the constant-per-use-case `trl_min`/`trl_max` columns)? Hand-labels the sample against a transparent keyword heuristic — 53% agreement, barely above the majority-band floor. Not viable as a plain keyword list. Read-only. |
| `venue_quality.ipynb` | Viability check for a candidate feature: looks up the 10 known-clean `venue` values (and a few known-dirty ones, as a negative-result check) against the OpenAlex `/sources` API, then joins the resulting venue-quality metrics (`works_count`, `2yr_mean_citedness`, `h_index`) onto their actual papers in `data/processed/papers_combined.parquet` to see whether external venue prestige diverges usefully from raw `citation_count` as a relevance signal, or just tracks it — it doesn't (r≈-0.04 with `triage_label`, r≈0.78 with the venue's own mean `citation_count`). Not viable. Read-only. |

## Running a notebook

Each notebook assumes it's run with its own folder as the working directory (so its
`../../data/raw`-style relative paths resolve) — open it from inside `notebooks/eda/`,
`notebooks/data_compile/`, or `notebooks/feature_experiments/`, not from `notebooks/`
itself.

## Conventions

- Every notebook explains *why*, not just *what*, in markdown cells — see the root
  `CLAUDE.md` for the project's non-negotiable conventions (NULL ≠ 0, cross-validate
  honestly, label which critic is speaking).
- A "Finding" markdown cell after a chart or table states what's a hard computed
  number versus a judgement call that needs more context to resolve — never leave a
  reader guessing which is which.
- A data-processing notebook (`data_compile/`) shows its inputs, raises loudly on
  anything unexpected instead of silently guessing (e.g. an unrecognised category
  value, a use case missing from a hardcoded registry), and prints the output
  metrics needed to trust the result before saving.
- A quickstart notebook stays plain — no custom color systems, no exotic diagnostics.
  It's meant to be the first thing a new contributor runs, not the most rigorous.
- Adding a new notebook? Add its folder (if it's a new category), a one-line
  description in the relevant table above, and — if it reads from `data/raw/` or
  writes to `data/processed/` — a note of what it reads/writes, same as the entries
  above.
