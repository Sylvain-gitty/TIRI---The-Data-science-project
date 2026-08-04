# Notebooks

Organised by what a notebook does, not by week — a notebook's folder tells you
whether it's safe to just read (`eda/`) or whether it writes to `data/processed/`
(`data_compile/`).

```
notebooks/
  eda/            exploratory analysis — read-only, no files written back to data/
  data_compile/   data-processing pipelines — data/raw/ -> data/processed/
  modelling/      Week-3 classifier-prep — fold/CV design, feature stacking; read-only
```

## eda/

| Notebook | What it does |
|---|---|
| `eda_quickstart.ipynb` | **Start here.** Loads `data/processed/papers_combined.parquet` and runs a basic, deliberately simple first pass — shape, dtypes, missing values, `.describe()`, key value counts, and a handful of plain charts. A launching-off point for anyone new to the project, not a deep dive. |
| `explore_use_cases.ipynb` | Compares structure and volume across the 6 labelled JSONL exports in `data/raw/` — schema, nulls, label balance, duplicates, citation/abstract-length distributions. Read-only. |
| `explore_usecase_definitions.ipynb` | Compares the 6 `usecase.json` search-brief definitions (problem statement, objective, terms, TRL constraints, decision criteria) against their matching JSONL export — structure, and a table joining each definition against its resulting corpus. Read-only. |

## data_compile/

| Notebook | What it does |
|---|---|
| `combine_use_cases.ipynb` | Reads all 6 JSONL exports + their matching `usecase.json` search-brief definitions from `data/raw/`, applies the cleaning/standardisation punch list found by the `eda/` notebooks (compound `paper_id`, nullable `Int64` dtypes, `venue` casing, `sources` parsed into booleans, fixed `triage_label`/`review_label` categoricals, `abstract_source` dropped, `has_abstract` flag), broadcasts both files' metadata onto every paper row, and writes `data/processed/papers_combined.parquet` — the file `scripts/compare_embeddings.py` and `scripts/train_baseline_classifier.py` are meant to consume via `--data`. See also `data/processed/README.md` for a full description of the output dataset. |

## modelling/

| Notebook | What it does |
|---|---|
| `wf_fold_pca_test.ipynb` | Week-3 classifier-prep on `papers_combined.parquet` (distinct from the diagnostic-only `scripts/compare_*.py`, per `HANDOFF.md`). Holds one use case out entirely for generalisation testing, builds a `StratifiedGroupKFold` scheme (stratified on use_case+label, grouped by first author) with explicit leakage checks, and tests a stacked ensemble (raw-embedding gradient boosting + PCA-reduced-embedding logistic regression) against a plain-embedding baseline, in-distribution and on the held-out use case. Read-only. |

## Running a notebook

Each notebook assumes it's run with its own folder as the working directory (so its
`../../data/raw`-style relative paths resolve) — open it from inside `notebooks/eda/`
or `notebooks/data_compile/`, not from `notebooks/` itself.

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
