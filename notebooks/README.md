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
| `explore_use_cases.ipynb` | Compares structure and volume across the 6 labelled JSONL exports in `data/raw/` — schema, nulls, label balance, duplicates, citation/abstract-length distributions. Read-only. |
| `explore_usecase_definitions.ipynb` | Compares the 6 `usecase.json` search-brief definitions (problem statement, objective, terms, TRL constraints, decision criteria) against their matching JSONL export — structure, and a table joining each definition against its resulting corpus. Read-only. |

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
- Adding a new notebook? Add its folder (if it's a new category), a one-line
  description in the table above, and — if it reads from `data/raw/` or writes to
  `data/processed/` — a note of what it reads/writes, same as the two entries above.
