# Notebooks

Three folders, one rule each:

| Folder | What's in it | Writes to `data/`? |
|---|---|---|
| **`main/`** | Two lines: the **11-notebook TIRI main line** (`01`–`11`), numbered in reading order — start here — and a parallel **4-notebook `*_benchset_v1` line** over the benchmark corpus, described in its own section below. | Yes — `01` and `04` of each line only |
| **`experiments/`** | Supporting evidence: viability checks, model bake-offs, superseded passes. Mostly **negative results**, kept deliberately. | Never |
| **`future_work/`** | Templates that have never been executed, by design. | Never |

Every notebook is committed **with its outputs intact**, so results are readable
without re-running anything.

---

## `main/` — the main line

Read in order. Each one states its own hard findings and its own scope limits.

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_data_compile.ipynb` | Reads all 6 JSONL exports + their matching `usecase.json` briefs from `data/raw/`, applies the cleaning punch list the EDA found (compound `paper_id`, nullable `Int64` dtypes, `venue` casing, `sources` parsed to booleans, fixed label categoricals, `has_abstract` flag), broadcasts each brief onto every paper row, and writes **`data/processed/papers_combined.parquet`**. Raises loudly on an unrecognised category or an unregistered use case rather than guessing. |
| 02 | `02_eda_quickstart.ipynb` | **The 5-minute tour.** Shape/dtypes first pass, then presentation-ready slides — corpus by label, each question's brief + label breakdown, missingness by feature × label, citation/age/author-count skew, and a PCA embedding map by question and by label. Each exported to `reports/eda_quickstart_*.{png,csv}` with printed takeaways. Deliberately plain — the first thing a new contributor runs, not the most rigorous. |
| 03 | `03_eda_full.ipynb` | The full EDA pass on `papers_combined.parquet`, targeting `triage_label`. Structure, class balance (including the finding that `pass` is 84.7% missing-abstract — a data-completeness artefact, not analyst hesitation, which is why `pass` is dropped as a modelling class), missingness, duplicates, univariate/bivariate stats, embedding separability (PCA + silhouette), and a leakage-safe `StratifiedGroupKFold` comparison of 3 candidate baselines. Adds a Mann-Whitney test on the citation-count/label claim (which **revises** the earlier eyeballed finding) and a paired significance test on the model ranking instead of eyeballing `std_auc`. **Hard finding:** Recall/F2 collapse far more violently than ROC-AUC on a held-out question (73–84% → 8–17% at threshold 0.5) — a threshold-calibration failure ROC-AUC alone doesn't show. |
| 04 | `04_feature_engineering.ipynb` | Assembles every validated, **row-local** feature block into one model-ready table: the Tier-1b lexical block (`scripts/lexical_features.py`), three embeddings joined separately (no concatenation, no PCA), cosine-to-brief + within-question rank per model, and admissible raw metadata. Nothing fitted (no PCA/scaler/imputer), so the output is safe to split into folds downstream. Writes **`data/processed/papers_fe.parquet`** (1,848 × 8,742). Needs `embeddings_cache/` — see the root README's data table. |
| 05 | `05_validation_design.ipynb` | **Where the central finding is measured.** Full Leave-One-Question-Out rotation across all 6 questions, with a cross-question duplicate-title leakage check, bootstrap CIs on each held-out AUC, and a train → validation → holdout decomposition separating the overfitting gap from the domain-shift gap. **Hard finding:** the collapse is general, not specific to one question (mean held-out AUC 0.50–0.54 for all 3 candidates); RandomForest/HistGradientBoosting score *below* the 0.5 dummy floor on 2 of 6 rotations each; the in-distribution ranking reverses under LOGO; and the domain-shift gap (~0.22–0.31 AUC) is consistently larger than the overfitting gap (~0.17–0.19). |
| 06 | `06_baseline_logreg.ipynb` | **The baseline, final and self-contained** (no dependency on `scripts/fold_pipeline_utils.py`, unlike the `experiments/` pipelines). A linear narrative: data health check → deterministic-only cleaning pre-split → `StratifiedGroupKFold` split (grouped by first author, stratified on question + label) → `StandardScaler`→`PCA(50, whiten)` on the embedding block, all fit **inside** the split → train → test → explicit scope limits. Scale-before-PCA and whitening are both required and documented. **Hard finding:** train→validation→holdout 0.906→0.871→0.867 AUC, no real validation→holdout gap (expected — the holdout is still drawn from the same 6 questions). PCA(50) retains 58.0% of embedding variance. |
| 07 | `07_advanced_catboost.ipynb` | **Advanced model.** `CatBoostClassifier` on the exact same split and feature space as 06, tuned for F2. Compares three hyperparameter search strategies (Grid, Randomized, Bayesian/Optuna-TPE) under an equal 24-evaluation budget; tunes the decision threshold on training-fold out-of-fold predictions, never on the holdout. **Hard findings:** the three search strategies land within 0.0024 of each other — at this budget, strategy choice barely matters; CatBoost's train→validation AUC gap (0.997→0.882) is over 3× the baseline's; and at threshold 0.5 it's a virtual tie with the baseline (F2 0.798 vs 0.790, AUC 0.864 vs 0.867 — CatBoost marginally *lower* on AUC). The large apparent win (F2 0.887 at threshold 0.2) is a **threshold effect**, not better ranking — AUC and AP are unchanged. |
| 08 | `08_ensemble_pooled.ipynb` | **Advanced model.** Stacks CatBoost (tuned for its ensemble role, a deliberately different objective from 07's standalone tuning), the 06 LogReg pipeline, and a separately-tuned RandomForest, via a LogisticRegression meta-learner on leakage-safe out-of-fold predictions. All three base models share the identical feature space, isolating model diversity from feature-access differences. **Every comparative claim is backed by a paired bootstrap test on the holdout, not a point-estimate table.** **Hard findings:** the ensemble beats the baseline significantly on AUC (+0.022, p=0.005) and F2 (+0.101, p<0.001), and beats standalone CatBoost on AUC (+0.025, p<0.001) — a genuine gain from stacking. But its F2 edge over standalone CatBoost (+0.005) is **not significant** (p=0.596, CI crosses zero). Also: judged at each model's own best threshold the three base models are within 0.005 of each other — the simple baseline is not meaningfully worse. |
| 09 | `09_ensemble_per_silo.ipynb` | **The production surface.** Runs `papers_fe.parquet` **per question, never pooled** (`CONTEXT.md` §1) — CatBoost + LogisticRegression on lexical + cosine-to-brief + metadata + the winning embedding, 50/50 averaged, walked through 7 reviewable stages: ablation, branch comparison, branch-disagreement correlation, ensemble-vs-branches, calibration, Recall@k/WSS@95, and a closing cold-start decision rule. CatBoost fits route through Modal — CatBoost's default `thread_count=-1` hits a confirmed severe slowdown on Apple Silicon. **Finding:** both branches clear the lexical+embedding LogReg baseline by ~0.03–0.18 ROC-AUC across questions; calibration sits close to the diagonal on 4/6 (the other 2 show real, quantified underconfidence). |
| 10 | `10_logo_ensemble.ipynb` | **LOGO, rerun against 08's ensemble instead of 05's plain baselines — nothing retuned.** Rotates 08's stacking model (CatBoost + LogReg + RandomForest → LogReg meta-learner) through 05's Leave-One-Question-Out fold structure, reusing 08's own fixed hyperparameters and decision threshold (0.2) verbatim. Reports train/validation/test F2 per rotation, plus a bootstrap omnibus test and Bonferroni-corrected pairwise post-hoc for whether test F2 differs across rotations. **Hard findings:** mean test AUC (0.697 ± 0.117) is far above 05's embedding-only baselines (~0.50–0.54), consistent with query-conditioned features being what lets a model transfer at all; the domain-shift gap is still larger than the overfitting gap, same pattern as 05; and the omnibus test rejects H0 (p≈0.0000) — test F2 genuinely differs across questions, resolving into three Bonferroni-corrected tiers. **Caveat stated explicitly in the notebook:** the top-scoring rotation by F2 (`solar_leo`) has the *worst* test AUC of the six — its high positive rate inflates F2 at this threshold regardless of ranking quality, so AUC is the fairer read of which questions actually generalize best. |
| 11 | `11_llm_benchset_a.ipynb` | **Can a prompted LLM replace any of this?** Runs four open-weights models (20B–397B) over `benchset_v1_large_set_a` — 8 externally-labelled collections, 62,229 papers, **2.19% relevant**, the first surface here where F2 discriminates (at our own 26–77% pools, "mark everything relevant" already scores 0.872). Scored on a case-control sample reweighted to population prevalence, gated before any spend by reproducing the on-disk cosine baseline both ways. Walks a brief-format ladder — the review's raw abstract, the supplied brief, a rule set **induced from 60 train labels**, and a deranged brief as falsification. **Findings:** zero-shot beats the free cosine baseline on 3 of 8 collections against a pre-registered ≥6/8, so it does not replace the cold-start rung; the model ranking scrambles completely versus our own pools, retiring "scale saturates at 31B" as a prevalence artefact; the F2-asymmetry prompt trick worth +0.35 at high prevalence is worth ±0.03 here; and an induced brief is worth **+0.057 ROC-AUC for one $0.006 call**, larger than the entire 20B→397B spread — but only at cold start, since folded into the trained ensemble it is worth +0.001. Shuffled-brief control collapses to AUC 0.498 / recall 0.000 on 8/8. |

**Why 08 and 09 both exist, and report different numbers.** 08 is the pooled
comparison — a random holdout drawn from the same six questions. 09 is per-question,
which is what would actually ship (a pooled model reads `use_case_key` off the embedding
at 96.2% accuracy, so it learns *which question this is*, not relevance). Their numbers
are measured on different surfaces and are **not comparable**. See the root README's
results table.

---

## `main/*_benchset_v1` — the same pass, at realistic prevalence

A parallel line of four notebooks over `data/benchsets_v1/` — 28 published
systematic-review screening collections, 181,199 papers, **1.86% relevant**. The six
questions above run 26–77% positive, so every threshold and F2 number measured on them was
measured ~20× away from production prevalence (`CONTEXT.md` §3). This is the corrective
surface.

They are numbered to match their TIRI counterparts but are **not ports** — the corpora
differ enough (28 silos not 6, no `embedding`/`relevance_score`/`venue` columns, a binary
label with no `pass`, NULL rather than empty-string abstracts, and a hard prohibition on
pooling) that several sections had to be replaced rather than adapted.

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_data_compile_benchset_v1.ipynb` | **Verification, not cleaning** — the corpus arrives de-duplicated with briefs broadcast, so instead of showing its own cleaning this notebook checks someone else's: 364 count/prevalence/per-column-coverage assertions against the corpus's `manifest.json`, plus its three headline figures re-derived rather than quoted. Adds `has_abstract` and `label_positive`, and writes **`data/processed/papers_benchset_v1.parquet`**. **Hard findings:** 3,137 papers were screened under more than one question and **130 come out `positive` under one and `negative` under another** — the first direct measurement of `CONTEXT.md` §2's central claim on external data; and a missing abstract here is a genuine NULL, the exact reverse of `papers_combined.parquet`, so carried-over `abstract == ""` code silently finds nothing. |
| 02 | `02_eda_quickstart_benchset_v1.ipynb` | The 5-minute tour, same five slides and palette as `02_eda_quickstart`, exported to `reports/eda_quickstart_benchset_v1_*`. Slide 1 shows the pooled bar **beside** the per-collection spread, because at 98% negative the pooled bar describes no collection in the corpus; the stacked per-question bar is replaced by log-scale size/positive-count panels for the same reason. **Hard findings:** prevalence spans 0.16%–78% (a 480× spread) and abstract coverage 0%–23% *by collection* — both invisible in any pooled view. |
| 03 | `03_eda_full_benchset_v1.ipynb` | The full pass, **every label-conditional statistic per collection**, win counts beside means, `roadfreight_metareview` out of every headline average. Embedding sections use Jasper + Qwen3-4B from `scripts/embed_benchsets.py`; the pooled O(n²) silhouette and pairwise-cosine of `03_eda_full` become per-collection and subsampled. §12 fits within-collection baselines on the 16 collections with ≥40 positives. **Hard findings:** citation count separates the classes in 24/27 collections and **23 point the same way — relevant papers are the *less* cited ones**, which is *not* what `03_eda_full` §7.1 found on TIRI and is best explained by reviews excluding the heavily-cited reviews and guidelines their search returns; and in 5 collections **publication year alone ranks above 0.70 ROC-AUC** (0.875 on `synergy_moran_2021`), a temporal artefact of the same kind `CONTEXT.md` §4 records for `solar_leo` — so §12 fits a year-only arm to keep it visible. |
| 04 | `04_feature_engineering_benchset_v1.ipynb` | Turns the pooled parquet into **three named datasets** — `benchset_v1_{small_test,large_set_a,large_set_b}.parquet` — under one inclusion rule (drop papers with no abstract: 2.6% of rows, 3.1% of positives) and two thresholds that are *demonstrated, not assumed*: the 40-positive bar is checked by running real `StratifiedGroupKFold` folds on every candidate, and the A/B split is found by enumerating all 16,384 partitions of the 15 fittable collections, minimising shared papers across the boundary. `large_set_a` is for iterating, `large_set_b` for touching once — both **within-silo**, since `CONTEXT.md` §1 rules out the cross-silo reading. Same feature contract as `04_feature_engineering`, embedding blocks included — `emb_jasper_*` / `emb_qwen4b_*` in the same wide-scalar layout, so the prefix-slicing in `06`–`10` works unchanged. That is 2.7 GB of the output and the reason it is not optional: those notebooks select with `startswith("emb_")`, which against a table lacking those columns returns `[]` and silently fits on metadata alone rather than raising. **Hard findings:** `synergy_walker_2018`'s abstract-less rows are **3.7× enriched for positives** (5.35% vs 1.45%), so this one rule makes the corpus's largest collection measurably *easier* than its real screening task — the caveat ships inside `reports/benchset_v1_split_manifest.json` rather than only in prose; and no balanced partition gets the leak to zero, so the **154 papers still sitting on both sides** are enumerated in `reports/benchset_v1_ab_crossing_papers.csv`, which turns the residual into a switch you can throw instead of a caveat you have to remember. |

**Run `scripts/embed_benchsets.py` before 02, 03 or 04.** It computes and caches the
Jasper-600M + Qwen3-Embedding-4B vectors those notebooks join, resumably, per collection.
04 reads it through `load_paper_vectors` / `load_brief_vectors` and bakes the vectors into
its three outputs, so the cache is a build-time dependency for them rather than a
permanent one — but it is what rebuilds them if a file is deleted.
`data/benchsets_v1/` itself is not tracked in git — see the root README's **Data** section.

---

## `experiments/` — supporting evidence

Read-only. This is where most of the project's measured **negative results** live, which
is why none of it was deleted.

**18 of the 20 below were executed from a fresh clone; 16 of those pass.** The two that don't fail
for different reasons, both about things deliberately kept out of git. (`wf_usecase_diversity.ipynb`
and `wf_brief_quality_detectors.ipynb` were added after that sweep and are not counted in it; both
execute top-to-bottom today.)

| Notebook | Blocker |
|---|---|
| `wf_synergy_validation` | `RuntimeError: OPENROUTER_API_KEY not set` — needs a key in `.env` (gitignored) |
| `wf_top_embeddings_generalization` | `FileNotFoundError` on `data/processed/embeddings_cache/` (~295 MB, gitignored) |

Both are committed with their outputs intact. `wf_embedding_model_bakeoff` **does** run —
it degrades gracefully, using the export's own embeddings and skipping the hosted and
GPU-side models it can't reach, rather than failing.

Three caveats on the 16 that pass:

- `author_orcid` (Crossref/ORCID) and `venue_quality` (OpenAlex) call **live APIs**, so
  they need network and their numbers can drift as those records change.
- `run_comparisons` and `wf_embedding_model_bakeoff` embed text from scratch, so they
  need `fastembed` **and** `sentence-transformers` actually installed — both are in
  `requirements.txt`, but they are its two heaviest entries (`sentence-transformers`
  pulls PyTorch) and are the easiest to skip on a partial install. First run also
  downloads the embedding models.
- `run_comparisons` is the slowest notebook in the repo for that reason.

### Feature viability checks

| Notebook | What it found |
|---|---|
| `terms_overlap.ipynb` | Per-paper term-overlap against its own question's must/nice/exclude lists. **Real signal (ROC-AUC 0.70–0.80) on 3 of 6 questions, none on the other 3.** The clue that led to the whole query-conditioned track — it was the only one of eleven metadata features that read the brief. |
| `terms_overlap_spacy.ipynb` | Does spaCy (lemma matching, phrase normalisation, negation-aware exclusions) beat plain regex? **Best variant gives +0.01 AUC on 2 of 3 working questions and rescues none of the 3 dead ones.** Marginal, not transformative — no gain for the cost. |
| `wf_feature_validation.ipynb` | Measures the 11-feature metadata punch list EDA recommended but never tested. **All 8 metadata columns together move out-of-fold AUC from 0.834 to 0.836.** Also shows nullness is 100% determined by which search API found the paper, that the percentile features are mathematically inert within a question, and that the "clean long venue strings" step would delete 40 real conference names to remove 2 dirty ones. |
| `venue_quality.ipynb` | External venue prestige (OpenAlex `works_count`, `2yr_mean_citedness`, `h_index`) vs raw citation count. **Not viable** — r≈−0.04 with the label, r≈0.78 with the venue's own mean citation count. It just tracks citations. |
| `trl_estimate.ipynb` | Can a paper's Technology Readiness Level be estimated from title + abstract via keywords? Hand-labelled 34-paper sample: **53% agreement, barely above the majority-band floor. Not viable.** |
| `author_orcid.ipynb` | Author prolificacy via ORCID lookup through Crossref. **20% coverage, publisher-dependent not prominence-dependent — not viable at that rate.** The headline result is the coverage rate, not the feature. |
| `wf_hard_use_cases.ipynb` | Why `solar_leo` and `soil_microbiome` resist every feature tried. Their classes are **interleaved in embedding space** (k-NN label agreement *below* the majority baseline), so no topical feature can separate them; `solar_leo`'s real boundary is publication vintage (a labelling-design property, not a missing feature). Also found the effect that outgrew the question: PCA-32 before combining is worth +0.03 to +0.08 on all six. |

### EDA and data understanding

| Notebook | What it does |
|---|---|
| `explore_use_cases.ipynb` | Structure and volume across the 6 raw JSONL exports — schema, nulls, label balance, duplicates, citation/abstract-length distributions. |
| `explore_usecase_definitions.ipynb` | The 6 `usecase.json` briefs against their matching exports — structure, and a table joining each brief to its resulting corpus. |
| `wf_data_enrich.ipynb` | Descriptive EDA on `papers_combined.parquet`: `notes` content, author/citation/venue breakdowns, outlier review, an interactive citations-vs-age scatter and a citations-by-label boxplot. |
| `wf_pre_pipeline_checks.ipynb` | Validates 4 open questions before pipeline work — `relevance_score` alone vs the embedding baseline, cross-question duplicate papers, abstract text-quality contamination, and whether term overlap is really just an abstract-length proxy (**it isn't** — length features alone reach 0.550, and removing them costs 0.004). |
| `sf_eda_firstrun.ipynb` | **v1 of the full EDA, superseded by `main/03_eda_full.ipynb`** — kept as a historical record. v2 resolved its binary-vs-3-class hedge, fixed a `StandardScaler`-fit-before-CV-split leakage bug, and added the significance tests v1 asserted without. |

### Embeddings and external validation

| Notebook | What it does |
|---|---|
| `wf_embedding_model_bakeoff.ipynb` | 11 candidate embedding models against the corpus's own local embedding — 7 hosted via OpenRouter, 4 heavier HF models run GPU-side via Modal — on one fold design, with a full metric bundle plus a check on whether combining embedding sources beats the best single one. |
| `wf_top_embeddings_generalization.ipynb` | Repeats the held-out test for the top 3 with **each** of the 6 questions held out in turn, tests prediction-level combination vs vector concatenation, and tests isotonic/Platt calibration under a simulated ~50-label budget. |
| `wf_synergy_validation.ipynb` | External validation against 3 [SYNERGY](https://github.com/asreview/synergy-dataset) systematic reviews — an independent benchmark this project had no hand in labelling, at realistic prevalence (1.7–14.8% positive vs our 26–77%). **Model ranking is not stable across prevalence regimes:** Qwen3-4B is mid-pack in-repo and *last* on SYNERGY. |
| `run_comparisons.ipynb` | Runs `scripts/compare_*.py`'s own functions with every table and plot inline, plus a paired significance check across representations. Set `USE_CASE_KEY` to analyse a different research question. Embeds the corpus from scratch, so it is the slowest notebook here and needs `fastembed` + `sentence-transformers`. |

### Spec and brief quality

| Notebook | What it does |
|---|---|
| `wf_usecase_diversity.ipynb` | Maps how far apart the 34 use cases sit (6 live + 28 benchset) and tests causally whether that diversity biases LOGO. §6 builds the brief × corpus matrix — score every use case's papers with every *other* use case's brief — which is where the foreign-brief margin comes from. **Hard finding:** `max_foreign_brief_auc` correlates with LOGO transfer at rho +0.70 but **+0.37 once prevalence is held constant**, and prevalence alone is −0.72; so nothing here may be read without controlling for it. |
| `wf_spec_reader_arm.ipynb` | Probe `P-R`: the same 14 brief variants the matcher arm scored, put in front of an LLM screener, to decide whether a spec linter needs one critic or two. **Hard findings:** contradiction, keyword padding, fluff and vagueness each cost the reader **less than the 0.03 noise floor** (three are *positive*), while a deranged brief costs **−0.282 AUC / −0.297 F2@own / −15.8pp fraction-read** — so the instrument is not blind. Stripping the brief to its bare topic name costs **−0.038**, which places the mechanism: no single field is load-bearing because the topic signal is **redundant** across fields. Every failure mode the reader sees, the matcher sees harder, so D43's two-critic linter collapses to one for *ranking* — but `only_name` raises fraction-read while lowering F2@own, which only a reader can show. $1.32 of a $10 ceiling; stage 2 unbought because the pre-registered stop rule fired. |
| `wf_brief_quality_detectors.ipynb` | Probes `P-FB` and `P-CK` from `reports/wf_spec_quality_plan.md`, both $0 and both label-free at scoring time. **Hard findings: both fail their pre-registered bars.** The foreign-brief margin is a property of the *embedding*, not the brief — rebuild the 34×34 matrix in BM25 space and 8 new collections flag that cosine called healthy (bar: ≤2), the two margins correlate at only rho 0.33, and `soil_microbiome`'s headline −0.190 becomes −0.021. The margin looked like a workload forecaster instead — rho **−0.929** against labelling gain — but that **fails on set B** (−0.464, CI crossing zero, sign flipping under prevalence control) and was never embedding-independent even on set A (qwen4b −0.571). Nothing ships. Checkability fails too: pooled IQR 0.055 against a 0.2 bar, `tech_forecasting` ranks 3rd of 6 when it should rank last, and its "stability check" compared a rulebook against itself (`checkability_broad` identical on all 34, max diff 0.0). |
| `wf_optimised_usecase_baseline.ipynb` | Rewrites TIRI's six specs **without ever seeing a label, a paper or the corpus** — every token comes from another field of the same file that no feature reads — then re-runs `main/06_baseline_logreg.ipynb` unchanged on each arm. **Hard findings:** the fitted baseline cannot tell the briefs apart (test F2 0.790 → 0.793, ROC-AUC 0.867 → 0.869, against ±0.017 / ±0.011 fold-to-fold spread *within* the unchanged baseline) — expected, because it is fitted on 24× the label budget at which brief quality stops mattering. At **zero labels** the enriched objective is worth **+0.014** mean ROC-AUC, clearing the 0.03 floor on 2 of 6 (`carbon_capture` +0.045, `solar_leo` +0.032); §3a decomposes that and the **objective carries +0.0135 of the +0.0139** — filling empty domain fields is worth +0.003 and is mildly *negative* on `solar_leo`. Topping up term lists averages nothing and costs `ner` **−0.116** from one added phrase, so `prose_only` is the shippable arm. Text volume does **not** predict the gain — the two biggest gainers grew least. §5 asks *where* the gain comes from: **not reduced spread** (sd widens on 2 of 3 encoders), **not** regression to the mean (gain vs baseline score rho −0.03), but concentrated on badly-**written** specs (`linter_findings` rho **+0.65**, LOO +0.45–0.79) — and 🔴 **the entire gain sits at 50–75% read depth, with nothing in the top 20%**, so no screening analyst would ever see it. ⚪ Also found: `lex_rank_bm25_must` is not bit-reproducible across processes (BM25 sums in set-iteration order; `PYTHONHASHSEED` flips 0–2 tied ranks of 1,848). |

### Fold and pipeline design

| Notebook | What it does |
|---|---|
| `wf_fold_pca_test.ipynb` | The original fold-design work: holds one question out entirely, builds the `StratifiedGroupKFold` scheme with explicit leakage checks, and tests a stacked ensemble against a plain-embedding baseline. `main/05_validation_design.ipynb` generalises this into the full 6-way rotation. |
| `sf_logo_fold_pipeline.ipynb` | The LOGO strategy rebuilt on `scripts/fold_pipeline_utils.py`'s config-driven `Pipeline`, scoped to LogisticRegression alone. **Still runs against the older `papers_combined.parquet` feature set**, not `papers_fe.parquet`. Reproduces the strategy notebook's numbers almost exactly (mean holdout AUC 0.536 ± 0.078 vs mean validation 0.760). |

---

## `future_work/` — never executed, by design

| Notebook | Status |
|---|---|
| `sf_catboost_fold_pipeline.ipynb` | **Template — `CONFIG["run_training"] = False`.** Every cell that would fit a model is gated and prints a skip message, so it's safe to Run All today. Superseded for actual results by `main/07_advanced_catboost.ipynb`; kept because it's the config-driven version. |
| `sf_llm_fold_pipeline.ipynb` | **Template — `call_llm_stub` raises `NotImplementedError` regardless of config.** A prompted-LLM classifier reading each question's brief plus the paper's title/abstract — no embedding, no training data, so it would work on a genuinely new question. §3 builds and prints one real prompt end-to-end with no model call, so the template is verifiably functional without training anything. **Not a result — a scaffold.** |

---

## Running a notebook

Every notebook resolves data via `../../data/...`, so **open it from inside its own
folder** (`notebooks/main/`, `notebooks/experiments/`, `notebooks/future_work/`), not
from `notebooks/` itself. All three sit at the same depth for exactly this reason.

Which ones run from a fresh clone, and which need regenerated data, is in the root
[`README.md`](../README.md#quick-start).

## Conventions

- Every notebook explains *why*, not just *what*, in markdown cells.
- A **"Finding"** cell after a chart or table states what is a hard computed number
  versus a judgement call — never leave a reader guessing which is which.
- A notebook that writes to `data/` shows its inputs, raises loudly on anything
  unexpected rather than silently guessing, and prints the metrics needed to trust the
  result before saving.
- Adding a notebook? Put it in the folder matching its role, add a row to the right
  table above, and note what it reads and writes.
