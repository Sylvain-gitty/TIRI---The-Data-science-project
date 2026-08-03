# Notebooks

Organised by what a notebook does, not by week — a notebook's folder tells you
whether it's safe to just read (`eda/`) or whether it writes to `data/processed/`
(`data_compile/`).

```
notebooks/
  eda/            exploratory analysis — read-only, no files written back to data/
  data_compile/   data-processing pipelines — data/raw/ -> data/processed/
```

## data_compile/

| Notebook | What it does |
|---|---|
| `combine_use_cases.ipynb` | Reads all 6 JSONL exports + their matching `usecase.json` search-brief definitions from `data/raw/`, applies the cleaning/standardisation punch list found by the `eda/` notebooks (compound `paper_id`, nullable `Int64` dtypes, `venue` casing, `sources` parsed into booleans, fixed `triage_label`/`review_label` categoricals, `abstract_source` dropped, `has_abstract` flag), broadcasts both files' metadata onto every paper row, and writes `data/processed/papers_combined.parquet` — the file `scripts/compare_embeddings.py` and `scripts/train_baseline_classifier.py` are meant to consume via `--data`. |

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
- Adding a new notebook? Add its folder (if it's a new category), a one-line
  description in the relevant table above, and — if it reads from `data/raw/` or
  writes to `data/processed/` — a note of what it reads/writes, same as the entries
  above.
