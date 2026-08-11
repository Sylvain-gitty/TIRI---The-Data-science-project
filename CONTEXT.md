# Project context — read this first

One primer for starting a new chat on either repo without re-explaining the project.
Built from a full read of every `.md` file in both `academic_research_agent` and
`TIRI---The-Data-science-project` (as of 2026-08-06). Scoped to **current state and the
reasoning behind it** — deep historical detail (an abandoned SaaS pivot, superseded
design revisions, ~7 not-yet-started spikes) is deliberately left out here; each section
below points at the source doc for anyone who needs that depth.

**People:** Sylvain (git `Sylvain-gitty`, initials `SF`/`sf`) and Warren (initials `wf`)
work across both repos. Warren owns/drives `academic_research_agent`; TIRI is their joint
data-science project on top of its exports — specifically, **TIRI is the "ensemble
project"** that repo's own `ONBOARDING.md` and `docs/ENSEMBLE_BRIEF.md` are written for.

---

## The two repos, in one sentence each

- **`academic_research_agent`** — a local-first tool that turns an analyst's use case
  into an active-learning literature-scouting loop: draft queries → retrieve across 7
  academic sources → embed + rank → keyboard-first triage → summarise → export.
- **`TIRI---The-Data-science-project`** — consumes that tool's labelled exports to build
  a classifier that predicts "is this paper interesting," so a future use case's
  screening queue can be prioritized and its accepted papers used to drive citation
  harvesting. **TIRI-only rule: never modify `academic_research_agent` from a TIRI
  session** — read it for reference, that's all.

---

## `academic_research_agent` — current state

**Stack & architecture:** React (Mantine) + FastAPI + Postgres/pgvector, local-first.
One `STEPS` constant drives navigation through 5 screens: Use case → Search → Triage →
Review → Data Analytics (Export is a header-action modal, not a spine step). The
marimo/DuckDB v1 app was deleted outright in the "v1 retirement" (2026-07-16) — Postgres
is the only engine now. `docs/ROADMAP.md` is the single live plan; `CLAUDE.md` is the
operational guide (DB safety, schema, conventions) — both are the sources of truth,
everything else is historical record or feeds into one of these two.

**Where it is on its own roadmap:** v1.3 (local-first, Warren + colleagues each on their
own clone). A hard gate blocks anyone from depending on a *hosted* instance (v1.4) until
durable jobs, rate limiting, and an LLM spend cap exist — deliberate, not an oversight.
An earlier attempt at a hosted multi-tenant SaaS (V2) was explored in depth (spiked,
priced, mostly de-risked technically) and then not pursued — free-API rate limits would
cap real multi-user concurrency without a paid tier that was never budgeted, and a
2026-07-11 decision to build locally first (v1.2, what exists today) made the hosted
track moot rather than formally killed. Full trail: `docs/archive/V2_SPIKE_RETRO.md`.

**Non-negotiable product principles** (load-bearing across every doc, including TIRI's):
- **`pass` is never a class.** It doesn't train anything, isn't a rejection — always
  exclude it from a binary target, never fold it into negatives.
- **Labels are append-only**, one label = one immutable fact; "undo" is relabelling, not
  deleting.
- **No model score reaches the pre-label triage deck** ("blindness before labelling") —
  a shadow LLM assessment (`paper_assessments`, shipped 2026-07-19) exists but is shown
  strictly post-label in Review, never before.
- **Scores are ranks, not probabilities.** Both the agent's own `relevance_score` and any
  LLM assessment need per-use-case calibration before a raw number means anything —
  TIRI's own work independently re-derived and hardened this into a hard rule (see below).
- **Retrieval has no LLM in it** — deterministic fan-out across sources, so
  precision@query and the query-performance ledger stay honest and reproducible.
- **NULL ≠ 0** everywhere — "never looked it up" and "looked, found zero" are different
  facts, never conflated.

**What Warren has identified as the two highest-leverage next moves** (his own
prioritization, `docs/STRATEGIC_REVIEW_2026-07.md`): (1) build a recall instrument —
today the app measures precision everywhere and recall nowhere, despite the product's
whole pitch ("scout earlier and more consistently") being fundamentally a recall claim;
(2) citation expansion for corpus growth — spiked and validated (10% seed recovers
~29% of held-out references, 5.6–18× over a matched null), schema drafted but not yet
applied to production. Both are framed as prerequisites for the ensemble project (TIRI),
not a detour from it.

**A separate, genuinely exploratory track** (concept/technology extraction — pulling a
technology graph out of the corpus) exists but nothing about it is committed; its own
framing is "get things wrong cheaply, nothing ships until a backtest gate passes," and
that backtest hasn't run yet. Don't treat any of its interim findings as production fact.

**Database safety (operational, not historical — carries over to any session touching
that repo):** `acagent` is Warren's real data — never point tests/seeds/UI at it, never
run destructive SQL on it. `acagent_smoke`/`acagent_test`/`acagent_e2e`/`acagent_firstrun`
are the safe databases for testing/QA. `make qa` is the one safe way to look at the real
app running against real-ish data.

**What TIRI needs FROM that repo that doesn't exist yet:** a `use_case_version` column on
exported rows (currently absent from both the JSONL header and `usecase.json`'s schema —
`usecase_schema_version` there is the *file format* version, not a spec-content version)
— needed so a future export where a use case's brief was refined mid-collection can be
told apart from one that wasn't. Currently a non-issue in practice (one embedding model
is constant across the whole combined dataset) but the pipeline should be built assuming
this will matter, not re-built once it does (`reports/wf_ensemble_report.md` §1).

**Deeper reading, only if needed:** `docs/decisions/` (10 ADRs, each with real reasoning
and rejected alternatives — 0004 append-only labels, 0007 "learn techniques not customer
terms," 0009 "validate the instrument before the verdict" are the three most likely to be
relevant to TIRI's own methodology); `spikes/README.md` (experiment registry — most
listed as "in progress" are actually unstarted paper-plans, only review_seeding,
query_tree, and citation-expansion have real completed findings).

---

## TIRI — current state

**The data:** `data/processed/papers_combined.parquet` — 2,873 papers across 6 real
research questions (`cement_binders` 690, `soil_microbiome` 602, `ner` 541, `solar_leo`
427, `carbon_capture` 323, `tech_forecasting` 290), each carrying its own real
objective/search-term brief, a `triage_label` (positive/negative/pass/never-reviewed),
and a precomputed 384-dim embedding (`paraphrase-multilingual-MiniLM-L12-v2`, constant
across every row today). Full data card: `data/processed/README.md`. Built by
`notebooks/data_compile/combine_use_cases.ipynb` from 6 raw JSONL exports +
`usecase.json` briefs in `data/raw/`.

**Target definition (resolved, not open):** binary `positive` vs. `negative` only.
`pass` is ~85% just "no abstract to triage," not a relevance judgement, and including it
as a 3rd class would let the model learn "text is missing" instead of "this paper is
relevant." `review_label` is unusable (99.97% null). The 478 never-triaged rows are the
eventual scoring pool, not training data.

**Three diagnostic scripts, explicitly NOT a model-selection pipeline** (`scripts/`,
each backed by a `reports/*_notes.md` decision trail): `compare_embeddings.py`,
`compare_ner_models.py`, `compare_combined_features.py` — test how a representation
relates to an already-labelled corpus (latent-space dispersion, use-case centrality,
classifier AUC, and a *query-conditioned* ranking AUC added later after a methodology
review found the classifier AUC alone can miss real problems — see
`reports/metrics_rework_and_rerun.md`). An earlier framing that treated one script's
"winner" as *the* Week-3 baseline model was deliberately retired — that coupling was
judged wrong, and the plumbing survives only as parked code in
`future_work/train_baseline_classifier.py`. Run `notebooks/comparisons/run_comparisons.ipynb`
to see any of this live; the CLI scripts write nothing to disk by default now.

**What real Week-2/3 investigation has actually found** (this is the substantive,
load-bearing work — full trails in `reports/wf_*.md`, `reports/sf_*.md`):

- **Feature engineering:** almost none of an 11-feature metadata punch list works
  (+0.002 AUC combined). Two are actively dangerous: `has_venue`/`citation_count_missing`
  turned out to be near-perfect proxies for *which search API found the paper*, not the
  paper's relevance — a model trained on them looks great in-distribution and worse on a
  genuinely new use case. **Term overlap** (does the paper contain the analyst's own
  must-include/nice-to-have words) is the one metadata feature with a real, measured
  effect, on 3 of 6 use cases. **The biggest lever isn't a new feature at all — it's
  compressing the 384-dim embedding to ~32 components (PCA) before concatenating
  anything to it.** This took held-out-use-case AUC from 0.520 (coin flip) to 0.714 and
  retroactively explains an earlier "combining features doesn't help" finding as an
  artifact of concatenating onto the *full-width* embedding, not a real verdict on the
  features themselves.
- **Two use cases resist every feature tried** (`solar_leo`, `soil_microbiome`) because
  their relevant/irrelevant papers are topically interleaved — no topical feature
  (embedding, TF-IDF, term overlap) can separate them. `solar_leo`'s real dividing line is
  publication year (a labelling-design artifact — the corpus was seeded from "canon,"
  then asked to reward papers that go beyond it). `soil_microbiome`'s real axis
  (applied-intervention vs. descriptive-ecology framing) lives in the use case's
  `objective` field, which no current feature reads.
- **Generalization is the headline problem, not in-distribution fit.** Leave-one-use-
  case-out testing (multiple independent notebooks, converging on the same result) shows
  every model tried — logistic regression, random forest, gradient boosting, stacking —
  collapses to ~0.50–0.59 AUC on a genuinely new use case, regardless of in-distribution
  score (which reaches 0.80+). **Recall/F2 collapse far more violently than AUC** on a
  held-out use case (73–84% in-distribution recall → as low as 8–17% held-out at a
  default threshold) — a calibration failure AUC alone hides.
- **`relevance_score` is disqualified as a model feature or baseline** — it's not a
  probability, not comparable across use cases (two different embedding models are live
  across the export), and drifts as the agent's own live refinement loop updates it. An
  earlier report crediting it with "generalizing better than embeddings" was a comparison
  error (apples-to-oranges) and has been retracted (`reports/wf_ensemble_report.md` §0).
- **Embedding model bake-off:** tested 11 candidate models (7 hosted via OpenRouter, 4
  GPU-hosted via Modal) against the current local baseline, with proper leave-one-use-
  case-out testing and external validation against the public SYNERGY benchmark
  (`github.com/asreview/synergy-dataset` — the same public dataset
  `academic_research_agent`'s own `evals/` also uses). **Recommendation: switch the
  corpus embedding to `Jasper-Token-Compression-600M` + `Qwen3-Embedding-4B`, concatenated**
  — real, validated improvement (0.585 → 0.63 mean AUC across all 6 use cases,
  confirmed on SYNERGY too), but **not adopted yet** and **not universal** —
  `carbon_capture` and `soil_microbiome` specifically didn't benefit. Isotonic
  calibration (not Platt — too unstable at small sample sizes) reliably fixes the
  out-of-domain score-meaning problem using ~50 labels from a new use case.
- **Baseline classifier shortlist** (not yet wired into an actual pipeline):
  `HistGradientBoostingClassifier` first (native missing-value handling, best raw score),
  `RandomForestClassifier` close second, `LogisticRegression` third but kept for
  calibration/interpretability — the gap between the top two is within one fold's own
  noise, don't over-read the ordering.
- **Fold discipline that's now standard across every notebook:** `StratifiedGroupKFold`,
  grouped by first author (prevents a prolific author's papers from straddling
  train/test), stratified on `use_case_key + label`. A real methodological trap was
  caught and documented: an early "grouping helps" reading was mostly a `shuffle=False`
  contiguous-block artifact, not a real grouping effect — always isolate one variable at
  a time when comparing CV schemes.

**Genuinely unstarted, per TIRI's own README:** the actual Week-3 modelling
pipeline — which features, which model, how to validate it — is a real, open decision,
deliberately not pre-decided by any of the diagnostic work above.

**Shared conventions across every script/notebook in TIRI** (all traceable to
`academic_research_agent`'s `CLAUDE.md`): NULL ≠ 0; cross-validated, never
train-then-score; "label which critic is speaking" (a hard number vs. a judgement call
must be visually/textually distinguishable); report honest negative results, not just
flattering ones — this repo's own trail includes several retracted or reframed earlier
claims, kept visible rather than quietly corrected.

---

## Open items a new session should know are still open

- **Whether to pool one model across all 6 use cases, or go per-use-case** —
  `academic_research_agent`'s `ENSEMBLE_BRIEF.md` posed this and TIRI hasn't settled it;
  the label-count-aware "adaptation ladder" in `reports/wf_ensemble_report.md` §4
  (unsupervised at 0 labels → calibration-only at ~50 → light adaptation at low hundreds
  → full retraining only at large stable pools) is the current best answer, untested.
- **Whether `solar_leo` should be excluded from a pooled training build** — its own brief
  admits the corpus is citation-biased canon; treating its numbers as evidence the system
  "finds relevant work" is explicitly flagged as a misuse.
- **Whether to adopt the embedding bake-off's recommendation** (switch + re-embed the
  whole corpus) — validated, priced (a few dollars), but not acted on yet.
- **The recall instrument** (`academic_research_agent` side) — flagged by Warren as the
  prerequisite for everything, including TIRI's own work; status of whether it's been
  built is not confirmed in any doc read.
- A "field guide" cross-referencing "11 open-source paper-scoring tools" is mentioned in
  `reports/combined_features_notes.md` as motivation for testing combined features, but
  that artifact itself wasn't found in either repo — likely exists elsewhere; flagging
  so a new session doesn't assume it's missing/lost, just not in scope of this read.

---

## Where to actually look for depth

| Question | Read |
|---|---|
| Why is TIRI structured this way, what happened before this file | `HANDOFF.md` (older, narrower — TIRI-internal branch history) |
| Embedding model choice reasoning | `reports/model_shortlist.md`, `reports/wf_embedding_bakeoff.md` |
| Feature engineering reasoning | `reports/wf_eda_fe_report.md`, `reports/wf_feature_plan.md`, `reports/wf_featureengineering_review.md` |
| Fold/generalization findings | `notebooks/modelling/`, `notebooks/pipelines/` |
| Baseline model choice | `reports/sf_baseline_classifier_shortlist.md` |
| What every notebook does | `notebooks/README.md` |
| Agent's architecture/conventions | `academic_research_agent/CLAUDE.md`, `AGENTS.md` |
| Agent's live plan | `academic_research_agent/docs/ROADMAP.md` |
| What TIRI needs from the agent's data-science colleagues | `academic_research_agent/docs/ENSEMBLE_BRIEF.md` |
