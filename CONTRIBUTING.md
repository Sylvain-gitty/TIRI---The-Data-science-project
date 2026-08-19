# Contributing to TIRI

Thanks for looking at this. Before anything else, two documents are worth more than this
one:

- **[`README.md`](README.md)** — what the project is, what it found, and how to run it.
- **[`CONTEXT.md`](CONTEXT.md)** — what is *true* here: the central finding, the open
  risks, and the **negative-results register** in §6. That register is the most valuable
  asset in the repo. Read it before proposing modelling work, because most well-intentioned
  suggestions on this project have already been measured and rejected, and the register says
  why.

This file is the *how*: the conventions the code already holds to, and what a pull request
is expected to clear.

---

## The conventions, and why they exist

These are not style preferences. Each one exists because breaking it produced a wrong
answer at least once on this project.

### 1. NULL is not 0

"We never found out" and "we looked and there was nothing" are different facts. A feature
over a sparse field emits `NaN` plus an explicit indicator column — it does **not** silently
`fillna(0)`.

Collapsing them destroys the distinction downstream and makes missingness look like a
measured zero. This rule is load-bearing in
[`scripts/lexical_features.py`](scripts/lexical_features.py) and
[`scripts/run_ensemble_candidate.py`](scripts/run_ensemble_candidate.py), and it is why the
EDA notebooks report missingness as a first-class result rather than a cleaning step.

### 2. Cross-validated, never train-then-score

Any accuracy / AUC / F2 number in this repo is measured on held-out folds. **A score on
data the model was fit on is not a result** and will not be accepted as one.

Anything fitted — scaler, PCA, imputer, threshold — is fitted *inside* the split, never
before it. `04_feature_engineering` deliberately writes an unfitted feature table for
exactly this reason: so the output is safe to split downstream.

### 3. Models are fit per research question, never pooled

This is the repo's central finding, not a preference. A pooled model recovers *which
research question a paper belongs to* from its embedding at 96.2% accuracy, so it learns
question identity rather than relevance, and leave-one-question-out performance collapses to
roughly chance. See [`CONTEXT.md`](CONTEXT.md) §1–2.

Consequences worth stating, because they are easy to get wrong:

- **Leave-one-question-out (LOGO) is a defaults-selection instrument, not a production
  estimate.** Never quote a LOGO score as "how well TIRI works".
- **Within-question (grouped k-fold inside one question) is the production surface.**
- Hyperparameter search happens **once, centrally**, in the LOGO loop — not per silo, where
  a few hundred labels would overfit.

### 4. Anything claiming to read the brief must survive a falsification control

Rebuild the feature against deliberately **wrong** briefs. If it still scores well, it is
measuring something generic and gets thrown away. This has already killed more than one
plausible feature.

`build_lexical_features(df, brief_map=...)` takes an injectable brief map for exactly this
reason. **Keep that seam in anything you add.**

### 5. Treat any gap below ~0.03 ROC-AUC as not established

Measured seed-to-seed noise on this corpus is **~0.010 ROC-AUC** and **0.015–0.027
WSS@95**. Six questions of 260–360 labelled rows is a small, heterogeneous sample, and it
has already produced at least one decision made on noise.

So: repeat across seeds and report the spread next to every number; report **per-question
win counts** alongside means, because a mean over six heterogeneous questions can be won by
being good at the easy ones; and never make an architecture decision from a single held-out
question.

Several things in this repo that look like wins are labelled ties for exactly this reason.
Please keep labelling them that way.

### 6. Label which critic is speaking

Where a hard metric sits next to a judgement call, say which is which. A reader cannot
otherwise tell a measurement from an opinion. Notebooks carry explicit **"Finding"** cells
for this, and the reports carry `Status:` lines.

### 7. One implementation, in `scripts/`

Shared logic lives in `scripts/` and is imported, not re-implemented — in particular
[`fold_pipeline_utils.py`](scripts/fold_pipeline_utils.py),
[`embedding_utils.py`](scripts/embedding_utils.py),
[`lexical_features.py`](scripts/lexical_features.py) and
[`benchset_metrics.py`](scripts/benchset_metrics.py).

Two corollaries the code already follows:

- **Model handling belongs in exactly one place**, so a change to how a model is invoked
  cannot land in one pipeline and miss another (see
  [`scripts/embed_benchsets.py`](scripts/embed_benchsets.py)).
- **Do not split Modal functions across files** — they are consolidated in
  [`scripts/modal_ensemble_experiments.py`](scripts/modal_ensemble_experiments.py) on
  purpose; that file's docstring explains what breaks otherwise.

### 8. Do not couple to the sibling repo's internals

TIRI consumes an export from a sibling tool (`academic_research_agent`, referred to in older
docstrings as `academic_agent`). That repo is **not** vendored here and is **not** on this
repo's path.

A `sys.path` hop across repos silently breaks the day the sibling moves or renames a private
symbol. Where sibling logic is genuinely needed, **copy it, byte for byte, with a dated
provenance note** — see [`scripts/checkability_patterns.py`](scripts/checkability_patterns.py)
for the pattern. Where a sibling *path* is needed, read it from an environment variable and
fail loudly if absent, as
[`scripts/run_checkability_audit.py`](scripts/run_checkability_audit.py) does.

---

## Documents referenced in this repo that are not in this repo

Older docstrings and reports cite a few documents by bare filename. They were internal or
sibling-repo documents and **are not part of this public repository**. If you hit one, this
is what it was:

| Cited as | What it was | Read instead |
|---|---|---|
| `CLAUDE.md` | The working-conventions file, used by the AI coding agents on this project. Its rules are the ones above. | **This file** |
| `DATA_BRIEF.md` | The sibling repo's provenance note for the benchmark corpus, including the "honest limits" list. Limit #2 — that set A's briefs derive from each review's own abstract, which inflates every brief-reading score — is quoted where it matters. | [`CONTEXT.md`](CONTEXT.md) §3, and the quoting report |
| `NUMBERS.md` | The product-side register of live figures (`N23`, `N33`, …). Where this repo says a numbered entry "should be removed", that is an instruction aimed at the sibling product, not at anything here. | [`CONTEXT.md`](CONTEXT.md) §6 |
| `academic_agent: .../PLAN.md` | Sibling-repo pre-registration documents. Already prefixed with the repo name where cited. | — |

These are left as-is in the historical reports rather than rewritten: those reports are a
dated decision trail, and silently editing their citations would misrepresent what was
written when. This table is the key.

---

## Working on the repo

### Setup

```bash
python -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # only for the NER comparison script
```

Install **all** of `requirements.txt`, not just the light half. `fastembed` and
`sentence-transformers` (which pulls PyTorch) are the two heaviest entries and the easiest
to skip — but anything that embeds text from scratch needs them, and the failure is a bare
`ModuleNotFoundError` several cells in rather than at import time.

For a byte-for-byte reproduction of the published numbers, use the pinned set instead:

```bash
pip install -r requirements-lock.txt
```

### Branches and commits

- Branches are prefixed with the author's initials: `wf-`, `sf-`. Notebook filenames carry
  the same prefix, so two work streams stay legible in a shared repo.
- **One logical change per commit.**
- Everything lands on `main` through a reviewed pull request.

### Notebooks

- Notebooks are committed **with their outputs intact**, so results are readable without
  re-running anything. Do not strip outputs.
- Notebooks assume they are run with **their own folder as the working directory**.
- Only `main/01` and `main/04` write to `data/`. Everything else is read-only, so notebooks
  can be run in any order once the data exists.
- `experiments/` never writes to `data/`. `future_work/` holds templates that have never
  been executed, by design — leave them unrun.

### Adding a seventh research question

The pipeline discovers files by glob and matches on the `use_case` name recorded *inside*
each file, not on the filename:

1. Drop `<new_key>.jsonl` and `<new_key>.usecase.json` into `data/raw/`.
2. Add a `USE_CASE_REGISTRY` entry in `notebooks/main/01_data_compile.ipynb`.
3. Re-run that notebook, then feature engineering.

The registry check raises loudly on an unregistered question rather than guessing a short
code. That is deliberate: the short code is what every downstream report will call it.

---

## What a pull request needs to clear

CI runs on every PR (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)). It is
deliberately modest — it checks that the repo is *coherent*, not that the science is right:

```bash
python -m compileall -q scripts/ future_work/   # everything parses
python -m pytest -q                             # the shared-library tests
ruff check .                                    # lint
```

CI does **not** execute notebooks. The light main-line notebooks take ~170 s combined and
the rest need up to 600 MB of untracked data, a GPU, or paid API keys, so notebook health is
verified by hand and recorded in the README's run table.

If your change touches a shared library in `scripts/`, please also run the four light
main-line notebooks (`01`, `02`, `03`, `05`) and confirm `01` still reproduces
`data/processed/papers_combined.parquet` unchanged.

### For a change that makes a claim

If your PR asserts that something works better, it needs the same standard as the rest of
the repo:

- The surface it was measured on, named explicitly (pooled / LOGO / within-question /
  external). These are **not comparable to each other** and conflating them is the easiest
  way to misread this project.
- Held-out numbers, not in-sample ones.
- Seed spread, and per-question win counts alongside any mean.
- An explicit note if the gap is inside the ~0.03 noise floor — which is not a reason not to
  report it, only a reason not to call it a win.

A negative result, clearly measured, is a welcome contribution here and belongs in
[`CONTEXT.md`](CONTEXT.md) §6.

---

## Reporting problems

Open an issue. If it is a reproducibility problem, please say which notebook or script,
which row of the README's run table you were following, and your Python and OS versions.

For anything security-related, see [`SECURITY.md`](SECURITY.md).
