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
  feature_engineering/  assembles validated feature blocks into one model-ready
                        dataset — data/processed/papers_combined.parquet ->
                        data/processed/papers_fe.parquet
  modelling/            Week-3 classifier-prep — fold/CV design, feature stacking; read-only
  pipelines/            config-driven fold + preprocessing pipelines, built to keep working
                        once feature engineering lands a differently-shaped dataset; read-only
  comparisons/    runs scripts/compare_*.py's own functions with output inline — reads
                  data/raw/, writes nothing (reports/ holds decision-trail .md only)
```

## eda/

| Notebook | What it does |
|---|---|
| `eda_quickstart.ipynb` | **Start here.** Loads `data/processed/papers_combined.parquet` and runs a basic, deliberately simple first pass — shape, dtypes, missing values, `.describe()`, key value counts, and a handful of plain charts. A launching-off point for anyone new to the project, not a deep dive. |
| `explore_use_cases.ipynb` | Compares structure and volume across the 6 labelled JSONL exports in `data/raw/` — schema, nulls, label balance, duplicates, citation/abstract-length distributions. Read-only. |
| `explore_usecase_definitions.ipynb` | Compares the 6 `usecase.json` search-brief definitions (problem statement, objective, terms, TRL constraints, decision criteria) against their matching JSONL export — structure, and a table joining each definition against its resulting corpus. Read-only. |
| `wf_data_enrich.ipynb` | Descriptive EDA on `data/processed/papers_combined.parquet`: `notes` content, author/citation/venue breakdowns by use case and `triage_label`, an outlier review (author/citation extremes, long venue names), author frequency per use case, and an interactive citations-vs-age scatter (plotly) plus a citations-by-label-by-use-case boxplot. Read-only. |
| `wf_pre_pipeline_checks.ipynb` | Follow-up to `reports/wf_eda_fe_report.md`: validates 4 open questions before pipeline/ensemble work — `relevance_score` alone vs. the embedding baseline (same fold scheme), cross-use-case duplicate papers (a risk for the held-out-use-case design), abstract text-quality contamination, and whether the term-overlap feature is really just an abstract-length proxy. Read-only. |
| `wf_embedding_model_bakeoff.ipynb` | Embedding model bake-off: evaluates 11 candidate models against the corpus's current local embedding — 7 hosted via OpenRouter (Qwen3-8B, OpenAI text-embedding-3-large, BGE-M3, Mistral, Gemini, Nemotron, Perplexity) plus 4 heavier HF models run GPU-side via the Modal app in `scripts/modal_embeddings.py` (SPECTER2, QZhou-Embedding, Jasper-Token-Compression-600M, Qwen3-Embedding-4B) — on the same `StratifiedGroupKFold` fold design + held-out use case as `wf_fold_pca_test.ipynb`, with a full metric bundle (ROC-AUC, PR-AUC, F2, Recall@k, Brier, latent-space dispersion/centroid) plus a check on whether combining embedding sources beats the best single one. Feature/embedding evaluation only, not the downstream classifier. Reads a local `data/processed/embeddings_cache/` (gitignored) to avoid re-calling paid APIs/GPU jobs on re-runs. |
| `wf_top_embeddings_generalization.ipynb` | Follow-up to `wf_embedding_model_bakeoff.ipynb`: repeats the held-out generalization test for Jasper/qwen3-8b/qwen3-4b with EACH of the 6 use cases held out in turn (not just `tech_forecasting`), tests combining Jasper+qwen3-8b at the *prediction* level (average / meta-learner on OOF probabilities, vs. the bake-off's vector-concatenation), and tests isotonic/Platt calibration under a simulated ~50-label per-use-case budget (`wf_ensemble_report.md` §4). Reuses cached embeddings, no new API/GPU cost. |
| `wf_synergy_validation.ipynb` | External validation against 3 [SYNERGY](https://github.com/asreview/synergy-dataset) systematic-review datasets (`Sep_2021`, `Menon_2022`, `van_der_Waal_2022`) — an independent benchmark this project had no hand in labelling, per `wf_ensemble_report.md` action item #2. Tests whether Jasper/qwen3-8b/qwen3-4b's win over the local baseline holds on completely different research domains and a different labelling process. |
| `sf_eda_firstrun.ipynb` | **v1, superseded by `sf_eda_v2.ipynb` below — kept as a historical record, not deleted.** Full EDA pass on `data/processed/papers_combined.parquet` targeting `triage_label` (`positive`/`negative`/`pass`, the "yes/no/pass" label): structure, target/class-balance analysis (including the `has_abstract`-confounds-`pass` finding, independently re-derived), missingness, duplicates, univariate + bivariate stats, embedding-space separability (PCA + silhouette), a feature-engineering decision table, and a leakage-safe (`StratifiedGroupKFold`, grouped by first-author) cross-validated comparison of 3 candidate baseline classifiers in both binary and 3-class framings, ending in a ranked top-3 baseline-model shortlist with merits. Builds on `wf_data_enrich.ipynb`, the `feature_experiments/` viability checks, and `wf_fold_pca_test.ipynb` rather than duplicating them. Read-only. |
| `sf_eda_v2.ipynb` | **v2 — start here instead of `sf_eda_firstrun.ipynb`.** Same EDA pass, revised after a critical-thinking review against this bootcamp's EDA/feature-engineering/hypothesis-testing curriculum. Resolves v1's binary-vs-3-class hedge (target is now `positive`/`negative` only, `pass` dropped as a modelling class — see §2.4); fixes a `StandardScaler`-fit-before-CV-split leakage bug in the baseline comparison; adds a Mann-Whitney significance test on the citation-count/label claim v1 asserted without testing (§7.1 — the result revises that finding); adds a paired significance test on the model ranking (§13.1, reusing `run_comparisons.ipynb` §4's method) instead of eyeballing `std_auc`; adds `recall_at_Xpct`/`wss_at_95` screening metrics (this repo's own convention, `scripts/embedding_utils.py`); and adds a held-out-use-case generalization check (§14) for the exact 3 candidates and feature set §13 uses, including a distributional comparison explaining *why* the gap exists, rather than citing `wf_fold_pca_test.ipynb`'s numbers by analogy. §13.2/§14.3 add Recall and F2 (`beta=2`) alongside ROC-AUC, and a train → validation → holdout decomposition. **Hard finding:** Recall/F2 collapse far more violently than ROC-AUC on the held-out use case (73–84% in-distribution recall → 8–17% held-out at the default 0.5 threshold) — a threshold-calibration failure ROC-AUC alone doesn't show. Read-only. |

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
| `wf_hard_use_cases.ipynb` | Diagnoses why `solar_leo` and `soil_microbiome` resist every feature tried so far. Finds their classes are **interleaved in embedding space** (k-NN label agreement below the majority-class baseline: −0.064 and −0.080), so no topical feature can separate them; that `solar_leo`'s boundary is publication vintage (the brief seeds the pool with cited canon, then rewards going "beyond the incumbent" — `year` 0.766 vs the embedding's 0.648), a labelling-design property rather than a missing feature; and that `soil_microbiome` splits on applied-intervention vs descriptive-ecology framing, a distinction stated in the use case's `objective` and absent from its `terms.must_include`. Also finds the effect that outgrew the question: adding a small feature to the raw 384-dim embedding is worth +0.001, but compressing to PCA32 first is worth +0.03 to +0.08 on all six use cases and takes held-out-use-case AUC from 0.520 to 0.714 — reframing `reports/combined_features_notes.md`'s "combined features don't help". Actions in `reports/wf_feature_plan.md` §5. Read-only. |
| `wf_feature_validation.ipynb` | Measures the 11-feature metadata punch list `reports/wf_eda_fe_report.md` §4 recommended from EDA but never tested (`citation_velocity`, `has_venue`, `venue_is_arxiv_only`, `author_count`, the two per-use-case percentile ranks, `is_english`). Builds each one, audits coverage/degeneracy, then reports per-use-case full-population ROC-AUC plus the decisive out-of-fold test on the `wf_fold_pca_test.ipynb` fold design: all 8 metadata columns together move OOF AUC from 0.834 to 0.836 on top of embedding + `relevance_score`. Also shows nullness is 100% determined by which search API found the paper, that the percentile features are mathematically inert within a use case, and that the "clean out long venue strings" step would delete 40 real conference names to remove 2 dirty ones. Verdicts written up in `reports/wf_feature_plan.md`. Read-only. |
| `venue_quality.ipynb` | Viability check for a candidate feature: looks up the 10 known-clean `venue` values (and a few known-dirty ones, as a negative-result check) against the OpenAlex `/sources` API, then joins the resulting venue-quality metrics (`works_count`, `2yr_mean_citedness`, `h_index`) onto their actual papers in `data/processed/papers_combined.parquet` to see whether external venue prestige diverges usefully from raw `citation_count` as a relevance signal, or just tracks it — it doesn't (r≈-0.04 with `triage_label`, r≈0.78 with the venue's own mean `citation_count`). Not viable. Read-only. |
| `sftestdropusecase.ipynb` | Ablation, not a feature check: combines both `notebooks/pipelines/` workflows (pooled and LOGO) into one notebook and drops `use_case_key` from every fold-stratification key (label-only stratification), to test whether use-case-aware stratification was helping or hurting — the classifier never saw `use_case_key` as a feature either way. **Finding:** no meaningful effect. LOGO's holdout AUC is bit-for-bit identical (0.536) since it never depended on inner-fold stratification; the pooled workflow's validation AUC barely moves (0.746 → 0.744) — the one place a bigger gap shows up (`final_holdout` AUC 0.753 → 0.793) is one single 20%-of-data holdout draw changing which specific rows landed in it, not a reproducible effect. Reuses `scripts/fold_pipeline_utils.py` unchanged. |

## feature_engineering/

| Notebook | What it does |
|---|---|
| `wf_build_fe_dataset.ipynb` | Assembles every already-validated, row-local feature block into one model-ready table: reads `data/processed/papers_combined.parquet` + the three cached embeddings in `data/processed/embeddings_cache/` (Jasper-Token-Compression-600M, Qwen3-Embedding-4B, Qwen3-Embedding-8B), filters to labelled rows, dedupes within each use case, adds the Tier 1b lexical block (`scripts/lexical_features.py`, `lex_*`), joins the three embeddings separately (`emb_jasper_*`/`emb_qwen4b_*`/`emb_qwen8b_*`, no concatenation, no PCA), adds cosine-similarity-to-brief + its within-use-case percentile rank per model, and admissible raw metadata (`year`, `paper_age`, `has_abstract`, `n_authors`, `citation_count`). Only row-local facts — nothing fitted (PCA/scalers/imputers) — so the output is safe to split into folds downstream. Writes `data/processed/papers_fe.parquet` (1,848 rows × 8,742 cols), the file `sf_*_fold_pipeline.ipynb` are meant to consume once their `CONFIG` is pointed at it. |

## modelling/

| Notebook | What it does |
|---|---|
| `wf_fold_pca_test.ipynb` | Week-3 classifier-prep on `papers_combined.parquet` (distinct from the diagnostic-only `scripts/compare_*.py`, per `HANDOFF.md`). Holds one use case out entirely for generalisation testing, builds a `StratifiedGroupKFold` scheme (stratified on use_case+label, grouped by first author) with explicit leakage checks, and tests a stacked ensemble (raw-embedding gradient boosting + PCA-reduced-embedding logistic regression) against a plain-embedding baseline, in-distribution and on the held-out use case. Read-only. |
| `sf_logo_fold_strategy.ipynb` | Generalises `wf_fold_pca_test.ipynb`/`sf_eda_v2.ipynb` §14's single-use-case holdout into a full Leave-One-Use-Case-Out (LOGO) rotation across all 6 use cases, on `sf_eda_v2.ipynb` §13's exact 3 baseline candidates + feature set, unchanged. Adds a cross-use-case duplicate-title leakage check, a bootstrap CI on each held-out AUC, Recall/F2 (`beta=2`) alongside ROC-AUC, and a train → validation → holdout decomposition (§3.2) separating the overfitting gap from the domain-shift gap. **Hard finding:** the collapse is general across all 6 use cases, not specific to `tech_forecasting` (mean held-out AUC 0.50–0.54 for all 3 candidates); `RandomForestClassifier`/`HistGradientBoostingClassifier` score *below* the 0.5 dummy floor on 2 of 6 rotations each; the in-distribution ranking (`HistGradientBoostingClassifier` > `RandomForestClassifier` > `LogisticRegression`) reverses on average under LOGO; the domain-shift gap (validation→holdout, ~0.22–0.31 AUC) is consistently larger than the overfitting gap (train→validation, ~0.17–0.19 AUC); and held-out Recall (7–20%) collapses far more than held-out AUC would suggest. Read-only. |

## pipelines/

Both notebooks below share `scripts/fold_pipeline_utils.py` (a `CONFIG` dict + a
`scikit-learn` `Pipeline`-building helper) instead of redefining preprocessing/metrics
code twice — see that module's own docstring for the "fit only on train" contract it
enforces. Both are built to keep working, after editing `CONFIG` only, once the
in-progress feature-engineering track ships a differently-shaped dataset — each notebook's
own §6 is a checklist for that swap, and §0 marks the one stand-in feature (`citation_velocity`,
not yet in `papers_combined.parquet`) each removes once the real dataset lands.

| Notebook | What it does |
|---|---|
| `sf_generalized_fold_pipeline.ipynb` | The "generalized" fold strategy: pools all 6 use cases and splits at the *row* level (`StratifiedGroupKFold`, grouped by first-author, stratified on a `use_case_key + label` composite) instead of holding a use case out — answers "how well does this LogisticRegression do on more data from research questions it has already seen," not "on a genuinely new one." Outer split carves a `final_holdout` (~20%), touched once; inner CV within the rest validates. **Hard finding:** validation AUC (0.746) and holdout AUC (0.753) — and Recall (0.723 vs. 0.714) — land within noise of each other here, unlike `sf_logo_fold_pipeline.ipynb`'s large validation→holdout gap; that contrast (a model can look "fine" under this design while still failing a genuinely new use case) is the point of building both notebooks side by side. |
| `sf_logo_fold_pipeline.ipynb` | The Leave-One-Use-Case-Out strategy from `sf_logo_fold_strategy.ipynb`, rebuilt on the same config-driven `Pipeline` architecture as `sf_generalized_fold_pipeline.ipynb`, scoped to `LogisticRegression` alone. **Hard finding:** reproduces `sf_logo_fold_strategy.ipynb`'s `LogisticRegression` numbers almost exactly (mean holdout AUC 0.536 ± 0.078, vs. mean validation AUC 0.760) — confirming the domain-shift gap (0.224) exceeds the overfitting gap (0.167) for this model alone, not just as an average across 3 models. Mean holdout Recall (0.196, down from 0.730 validation) is propped up almost entirely by one rotation (`cement_binders`, 0.606); the other five sit at 0.011–0.238. |
| `sf_catboost_fold_pipeline.ipynb` | **Template — `CONFIG["run_training"] = False`, not executed.** Both fold strategies above, rebuilt around `CatBoostClassifier` via `scripts/fold_pipeline_utils.py`'s `build_tree_pipeline`/`build_tree_preprocessor` (no scaling/imputation — native missing-value + `category`-dtype categorical handling instead; `boosting_type="Ordered"` and `cat_features` pinned explicitly). Every cell that would fit a model is gated and prints a skip message; safe to Run All today. Picked over LightGBM/XGBoost specifically because ordered boosting targets the target-leakage-during-training already observed on this dataset (`RandomForestClassifier`/`HistGradientBoostingClassifier` both hit train AUC 1.000 under LOGO) — not assumed to close the domain-shift gap on its own; see this session's model-recommendation writeup for the full reasoning. |
| `sf_llm_fold_pipeline.ipynb` | **Template — `CONFIG["run_training"] = False`, and `call_llm_stub` raises `NotImplementedError` regardless.** A prompted-LLM classifier that reads each use case's own broadcast brief (`objective`, `terms_must_include/nice_to_have/exclude`, `decision_rules`, ...) plus the paper's title/abstract — no embedding, no scaling, no training data required for a genuinely new use case. Uses the new `scripts/llm_pipeline_utils.py` for prompt construction, leakage-safe few-shot example selection (drawn from the training fold only, never validation/holdout), and response parsing; reuses `fold_pipeline_utils.py`'s split/metric helpers unchanged. §3 builds and prints one real prompt end-to-end with no model call, so the template is verifiably functional without training or testing anything. |
| `wf_ensemble_fold_pipeline.ipynb` | Phase 2 of the ensemble-modelling plan (Phase 1: `scripts/modal_ensemble_candidate.py` → `reports/wf_ensemble_v1_candidate.md`). Runs `data/processed/papers_fe.parquet` **per use case, never pooled** (`CONTEXT.md` §1) — CatBoost + LogisticRegression on Tier-1b lexical + cosine-to-brief + metadata + the winning raw embedding (picked programmatically from Phase 1's numbers), 50/50-averaged, walked through 7 reviewable stages: ablation, branch comparison, branch-disagreement correlation, ensemble-vs-branches, calibration, Recall@k/WSS@95, and a closing cold-start decision rule. CatBoost fits call the deployed `tiri-ensemble-ablation` Modal app rather than fitting locally — CatBoost's default `thread_count=-1` hits a confirmed severe slowdown on this development machine's Apple Silicon. **Finding:** the two embedding treatments (Qwen3-8B vs. Jasper+Qwen3-4B concat) are a near-tie as expected, both branches clear the Tier1b+embedding LogisticRegression baseline by ~0.03-0.18 ROC-AUC across use cases, and calibration is close to the diagonal on 4/6 use cases (2/6 show real, quantified underconfidence, not just visual bowing). |

## comparisons/

| Notebook | What it does |
|---|---|
| `run_comparisons.ipynb` | **The primary way to run and see this repo's comparisons.** Imports `scripts/compare_embeddings.py`, `compare_ner_models.py`, `compare_combined_features.py`'s own functions (doesn't reimplement them) and shows every table and plot inline, plus a paired significance check across representations — clone the repo, run all cells, see the full result with nothing pre-generated. Toggle `CORPUS_KEY` to switch between the two real corpora. Writes nothing to disk. See `reports/metrics_rework_and_rerun.md` for the evaluation-metrics rework this notebook exercises. |

## Running a notebook

Each notebook assumes it's run with its own folder as the working directory (so its
`../../data/raw`-style relative paths resolve) — open it from inside `notebooks/eda/`,
`notebooks/data_compile/`, `notebooks/feature_experiments/`, `notebooks/feature_engineering/`,
`notebooks/modelling/`, or `notebooks/pipelines/`, not from `notebooks/` itself.

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
