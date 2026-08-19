<!--
Keep this short. The repo's conventions are in CONTRIBUTING.md; this template is only
here so a reviewer does not have to ask the same four questions every time.

Delete any section that genuinely does not apply.
-->

## What this changes

<!-- One or two sentences. What is different after this lands? -->

## Why

<!-- What problem or question prompted it. Link the report or issue if there is one. -->

---

## If this PR makes a claim

Skip this whole section for a docs, tooling, or refactor PR. Fill it in if you are asserting
that something works, works better, or does not work.

- **Surface measured on:** <!-- pooled / LOGO / within-question / external benchmark -->
  <!-- These are NOT comparable to each other. Naming the wrong one is the easiest way to
       misread this project — see README § Results. -->
- **Held-out numbers:** <!-- not in-sample -->
- **Seed spread:** <!-- measured noise here is ~0.010 ROC-AUC, 0.015–0.027 WSS@95 -->
- **Per-question win count:** <!-- e.g. 4 of 6 — a mean over six heterogeneous questions can
                                   be won by being good at the easy ones -->
- **Is the gap inside the ~0.03 noise floor?** <!-- yes/no. "Yes" is fine and reportable;
                                                    it just is not a win. -->

If the change adds or modifies a feature that reads the brief, confirm the falsification
control:

- [ ] Rebuilt against deliberately wrong briefs, and it does **not** still score well
      (`CONTRIBUTING.md` §4)

---

## Checks

- [ ] `python -m compileall -q scripts/ future_work/` passes
- [ ] `python -m pytest -q` passes
- [ ] `ruff check .` passes
- [ ] No `fillna(0)` over a sparse field — NULL is not 0 (`CONTRIBUTING.md` §1)
- [ ] Nothing fitted outside its fold (scaler / PCA / imputer / threshold)
- [ ] No new hardcoded absolute path; sibling-repo paths come from an env var
- [ ] Notebooks committed **with outputs**, and no notebook output contains a local
      filesystem path

### If a shared library in `scripts/` changed

- [ ] Ran the four light main-line notebooks (`01`, `02`, `03`, `05`)
- [ ] `01` still reproduces `data/processed/papers_combined.parquet` unchanged

## Anything a reviewer should push back on

<!-- Optional, and genuinely useful. Name the weakest part of this PR yourself — a caveat you
     would rather state than have found. The repo does this to itself throughout. -->
