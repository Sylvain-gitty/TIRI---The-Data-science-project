# Notebooks

Organised by what a notebook does, not by week — a notebook's folder tells you
whether it's safe to just read (`eda/`) or whether it writes to `data/processed/`
(`data_compile/`).

```
notebooks/
  eda/            exploratory analysis — read-only, no files written back to data/
  data_compile/   data-processing pipelines — data/raw/ -> data/processed/
```

## eda/

| Notebook | What it does |
|---|---|
| `eda_quickstart.ipynb` | **Start here.** Loads `data/processed/papers_combined.parquet` and runs a basic, deliberately simple first pass — shape, dtypes, missing values, `.describe()`, key value counts, and a handful of plain charts. A launching-off point for anyone new to the project, not a deep dive. |

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
