# Reports — the decision trail

57 write-ups covering what was tried, what was measured, and what was rejected. They were
written over four weeks and **they do not all still hold**. This index is the map; every
report in the folder is listed somewhere below.

**On `Status:` lines.** 13 of the 57 carry one. The convention started partway through and
was never applied retrospectively, so the absence of a `Status:` line means nothing — read
the section heading here instead, which is where the current/historical/superseded judgement
actually lives for every report.

A note on what those judgements mean — and which critic is speaking. `current` /
`historical` / `superseded` are **navigational judgements made when tidying the repo**,
not measurements. The numbers inside each report are unchanged and remain what they
were when measured; what these labels tell you is whether a later document revised the
*conclusion* drawn from them.

---

## Start here

| Report | What it is |
|---|---|
| [`tiri_whitepaper.md`](tiri_whitepaper.md) | **The whole project as one narrative document**, written for a reader who has not seen the repo: the problem, why F2, the data, the finding that reshaped the work, what ships, the external validation, and appendices covering the metrics in plain English and the rejected list with measurements. Start here if you want the argument rather than the decision trail. Renders to `.docx` with `npm run whitepaper`. |
| [`wf_ensemble_final_recommendations.md`](wf_ensemble_final_recommendations.md) | **The single decision doc.** Every architecture choice, its evidence, and a confidence grade — plus an explicit "do not re-propose these" list. If you read one file in this folder, read this. |

## Spec and brief quality — the 2026-08-13/14 block

| Report | Answers |
|---|---|
| [`wf_spec_quality_answer.md`](wf_spec_quality_answer.md) | **Start here for the spec-quality question.** Consolidates all six runs into the answer: the Minimum Viable / Optimal / Never spec with paired set-A/set-B evidence, and the verdict on scoring. Three fields carry the value (`objective` +0.151/+0.263, `terms_nice_to_have` +0.105/+0.215, `terms_must_include` by count), three are **decoration to every consumer on both surfaces** (`problem_statement`, `domain_*`, `terms_exclude`). **No single score is possible** — three attempts, three failures, and the reason is that prose damage and term damage invert between label regimes. What is buildable: a priced label-free linter, plus the shuffled-brief control as a validated "is it done" test. |
| [`wf_spec_quality_plan.md`](wf_spec_quality_plan.md) | **Read this first for any of the four rows below.** The pre-registration, plus a dated Amendment recording five false premises corrected *before* any probe ran — three of the four probes had bars that could not be read as written. Every bar lives here, not in the findings. |
| [`wf_spec_quality_ablation.md`](wf_spec_quality_ablation.md) · [`_set_b`](wf_spec_quality_ablation_set_b.md) | What each brief field is worth to a **matcher**, and what the four classic bad specs cost. Set A first, then set B as a clean check — where **most of the term-damage finding does not replicate**: `keyword_flood` −0.072→−0.031, `no_must` −0.043→−0.010. Set A's means were carried by one collection. What does replicate: contradiction barely dents a matcher (−0.014→−0.009). |
| [`wf_spec_quality_reader.md`](wf_spec_quality_reader.md) | The same 14 texts against a **reader** (LLM). The two arms do not disagree the way S-AL predicted. **No single brief field is load-bearing for a reader, because the topic signal is redundant across fields** — every localised degradation lands inside the noise floor, stripping the brief to its topic name costs only −0.038, and a wrong brief costs −0.282. Read the shuffled-brief floor beside every number or the null is unreadable. |
| [`wf_foreign_brief_detector.md`](wf_foreign_brief_detector.md) | Is "a stranger's brief screens your corpus better than your own" a brief defect? **No — it is a property of the embedding.** Rebuilt in BM25 space, 8 new collections flag that cosine called healthy. The fallback reading — that it forecasts labelling cost — **also fails on set B** (rho −0.464, CI −1.000 to +0.765, sign flipping under prevalence control). `NUMBERS.md` N33 ships this caption today and should be **removed**. |
| [`wf_foreign_brief_validity_setb.md`](wf_foreign_brief_validity_setb.md) | The confirmatory read `benchset_v1_large_set_b` was reserved for, and it **kills** the foreign-brief margin's fallback caption. rho −0.929 on burned set A becomes **−0.464 (CI −1.000 to +0.765)** on clean data, with the sign **flipping to +0.143** under prevalence control — and it was never embedding-independent even on A (qwen4b −0.571, CI crossing zero). Also the methodological note: leave-one-out stability tests whether one *point* carries a correlation, never whether the whole *surface* does. |
| [`wf_spec_linter.md`](wf_spec_linter.md) | **The answer, as a runnable tool.** `scripts/spec_linter.py` — 10 deterministic, label-free checks, each carrying the measured cost that prices it, quoted on three surfaces. No score, by design (see the answer §5). Calibrated as a test: `--self-test` asserts every check fires on the exact string whose cost was measured (imported, never restated) and that **no check fires spuriously on any of the 34 real specs**. Run on those 34 it returns **5 findings, all on TIRI's own use cases, 0 on benchset** — `solar_leo` has one nice-to-have term where one term recovers only a quarter of the field's value. |
| [`wf_checkability_audit.md`](wf_checkability_audit.md) | Can we say a use case is automatable before scoring it? **No.** Pooled IQR 0.055 against a 0.2 bar; `tech_forecasting` ranks 3rd of 6 when it should rank last. Keeps two findings that are not the metric: half of TIRI's own use cases never state what evidence they want, and half this metric's own definition never fired. |
| [`wf_optimised_usecase_baseline.md`](wf_optimised_usecase_baseline.md) | **What the whole spec-quality line is worth in production terms.** Applies the answer's advice to TIRI's six specs — leak-free, every token from a field no feature reads — and re-runs `06_baseline_logreg.ipynb` unchanged. **The fitted baseline cannot see it** (test F2 +0.003, ROC-AUC +0.002, against ±0.017/±0.011 spread *within* the unchanged baseline), which is the predicted result at 24× the label budget where brief quality stops mattering. At **zero labels** the enriched objective is worth **+0.014** mean, **+0.045** on `carbon_capture`, clearing the floor on 2 of 6; the objective carries **+0.0135 of the +0.0139** and filling empty domain fields carries the rest. Term-list padding averages nothing and costs `ner` **−0.116**. Two harness bugs were caught by instrument checks before any result was read, one of them **five times larger than the effect being measured**. |
| [`wf_optimised_gain_source.md`](wf_optimised_gain_source.md) | *Where* that gain comes from. **It is not reduced spread** — sd across the six use cases is flat on one encoder and *widens* on two, and the worst-scoring use case goes backwards on two of three. It **is** concentrated on badly-**written** specs (`linter_findings` rho **+0.65**, leave-one-out +0.45 to +0.79; `checkability` +0.54; term count +0.54) and has **no relationship to which use case scored badly** (rho −0.03), which rules out regression to the mean. 🔴 **The finding that outranks both: the entire gain sits at 50–75% read depth.** Change in recall at 5/10/20% depth is −0.000 / −0.003 / −0.001. It is real, measurable, and in the part of the list nobody reads. |
| [`wf_benchset_rewrite_feasibility.md`](wf_benchset_rewrite_feasibility.md) | **Read before proposing the benchset confirmation again — it was tried and it cannot be run.** Two independent blockers: `spec_linter.py` returns **0 findings on all 28**, so the predictor that carried the n=6 result is *constant* there (and `n_terms` has 1/7th of TIRI's spread — the 28 are machine-generated and uniform by construction); and their rewrite reservoir is **empty on 0/28**, while the only populated substitute is stamped `drafted_from: "review_abstract+labels"` and carries `exemplars` — the titles of accepted papers. So the n=6 caveat on "the linter predicts where a rewrite pays" is the **ceiling, not a to-do**. ⚪ Also fixed two linter bugs this exposed: a crash on any clean run, and `--specs benchset` silently overwriting the shipped all-34 report. |

## Current — the live evidence

| Report | Answers |
|---|---|
| [`wf_query_conditioned_findings.md`](wf_query_conditioned_findings.md) | Why leave-one-question-out collapses, and the brief-reading feature block that responds to it. The narrative behind the project's central finding. |
| [`wf_tier1b_lexical_control.md`](wf_tier1b_lexical_control.md) | The Tier-1b lexical block and its **shuffled-brief falsification control** — the pattern to copy for any feature claiming to read the brief. |
| [`wf_synergy_recall_validation.md`](wf_synergy_recall_validation.md) | External validation at realistic prevalence (1.7–14.8% positive) on reviews we had no hand in labelling. The least self-graded evidence here. |
| [`wf_embedding_recall_comparison.md`](wf_embedding_recall_comparison.md) | Jasper / Qwen3-4B / Qwen3-8B judged on WSS@95, on both fold surfaces. |
| [`wf_featureengineering_review.md`](wf_featureengineering_review.md) | The 11-feature metadata punch list, measured: +0.002 AUC in total, and harmful out-of-domain. |
| [`wf_feature_plan.md`](wf_feature_plan.md) | The per-feature verdicts from that measurement. |
| [`metrics_rework_and_rerun.md`](metrics_rework_and_rerun.md) | The evaluation-metrics rework. **Read before trusting any ROC-AUC in a report older than it.** |
| [`model_shortlist.md`](model_shortlist.md) | Decision trail for the embedding-comparison *diagnostic* (not the production choice). |
| [`ner_model_notes.md`](ner_model_notes.md) | Decision trail for the NER comparison — and why no NER feature ships. |

## Current, supporting — evidence for the ensemble

| Report | Role |
|---|---|
| [`wf_ensemble_v1_candidate.md`](wf_ensemble_v1_candidate.md) | Ensemble v1 Phase 1 — the feature/embedding/branch ablation. |
| [`wf_ensemble_v2_multiseed_verification.md`](wf_ensemble_v2_multiseed_verification.md) | The multi-seed, noise-floor-gated re-check of the v2 experiments. |
| [`wf_lean_vs_embedding_arms.md`](wf_lean_vs_embedding_arms.md) | Does the ensemble need the raw embedding block? |
| [`wf_ensemble_v2_experiments.md`](wf_ensemble_v2_experiments.md) | **Exploratory, single-seed.** Directions, not final numbers — check the multi-seed verification and the final recommendations before quoting anything from it. |

## Historical — accurate when written, conclusions since revised

| Report | What changed |
|---|---|
| [`wf_ensemble_report.md`](wf_ensemble_report.md) | Pre-dates Ensemble v1. Superseded on architecture, but still the source of record for two negative results (`relevance_score`; per-customer hyperparameter search). |
| [`wf_eda_fe_report.md`](wf_eda_fe_report.md) | Its §4 punch list was proposed from EDA alone, then measured and mostly rejected. The hypothesis, not the result. |
| [`wf_embedding_bakeoff.md`](wf_embedding_bakeoff.md) | Superseded on embedding choice. Several Round 1–2 numbers rest on one held-out use case — see `CONTEXT.md` §5. |
| [`combined_features_notes.md`](combined_features_notes.md) | Its "combining made things worse" headline did not survive significance testing (p=0.475–0.673). Read as "no measurable effect". |

## Superseded — kept for the reasoning trail

| Report | Replaced by |
|---|---|
| [`wf_ensemble_v1_results.md`](wf_ensemble_v1_results.md) | [`wf_ensemble_final_recommendations.md`](wf_ensemble_final_recommendations.md) |
| [`sf_baseline_classifier_shortlist.md`](sf_baseline_classifier_shortlist.md) | `notebooks/main/06_baseline_logreg.ipynb` (its source notebook was the v1 EDA) |

---

## `benchset_v1` — the benchmark corpus, kept separate on purpose

Everything below comes from `data/benchsets_v1/` — 28 published systematic-review
screening collections, 181,199 papers, **1.86% positive** — *not* from the six TIRI
questions the rest of this folder is about. The two corpora are never averaged together
and a number from one does not transfer to the other, which is why every filename here
carries a `benchset_v1` prefix and why they sit in their own section.

The distinction that matters: the TIRI pools run **26–77% positive**, roughly 20×
production prevalence (`CONTEXT.md` §3). These files are the prevalence-realistic
counterpart. A threshold, an F2 or a WSS@95 measured on one is not comparable to the
same statistic measured on the other.

| Artifact | What it is |
|---|---|
| [`benchset_v1_split_manifest.json`](benchset_v1_split_manifest.json) | **The decision record for the three benchmark sets.** Row and positive reconciliation at each filtering step, both derived thresholds, the A/B partition search and what it cost, and the caveats that must travel with the data — including that `synergy_walker_2018` is easier here than its real screening task. Written by `notebooks/main/04_feature_engineering_benchset_v1.ipynb`. |
| [`benchset_v1_ab_crossing_papers.csv`](benchset_v1_ab_crossing_papers.csv) | The 154 papers (313 rows) that sit in both `large_set_a` and `large_set_b` because no balanced partition could separate them. Filter `set_b` on `row_key` for a strict held-out read — the residual leak as a switch, not a caveat. |
| `eda_quickstart_benchset_v1_*.{png,csv}` | The five quickstart slides for this corpus, counterparts to `eda_quickstart_*`. Same slides, different corpus — do not read them side by side as a before/after. |

The narrative for all of it is in the four `*_benchset_v1` notebooks, not in a write-up
here: this corpus arrived after the reports above were written, and its findings live
in the notebooks that measured them.

## Why nothing was deleted

The negative results are the point. `CONTEXT.md` §6 keeps a register of everything
measured and rejected — the 11-feature punch list, TRL keyword estimation, venue
quality, ORCID lookup, spaCy over regex, prediction-level stacking, PCA-64 within a
silo — precisely so nobody spends a week re-running them. A superseded report is still
the evidence for why something was tried and dropped; it just needs a label saying so,
which is what the `Status:` lines are for.

## LLM screening — the pilot on TIRI, then set A at 2.19% prevalence

The two decision docs are **`wf_llm_pilot_findings.md`** (TIRI's own pools) and
**`wf_llm_benchset_a_findings.md`** (the external benchmark). Everything else in this section
is the evidence behind them. Read the two findings docs in that order; the benchmark run
retired several of the pilot's conclusions, and the reason is prevalence.

| Report | What it is |
|---|---|
| [`wf_llm_screening_plan.md`](wf_llm_screening_plan.md) | The pre-registration. Every bar lives here, including the ≥6/8 zero-shot bar the run then failed. |
| [`wf_llm_pilot_findings.md`](wf_llm_pilot_findings.md) | **The pilot's decision doc.** A prompted LLM does not replace the ensemble, and is rejected as a third branch. |
| [`wf_llm_pilot_results.md`](wf_llm_pilot_results.md) | The pilot's raw results table. |
| [`wf_llm_split_results.md`](wf_llm_split_results.md) | F2 across train / test / validate, all six questions pooled. |
| [`wf_llm_branch_experiment.md`](wf_llm_branch_experiment.md) | Does a prompted-LLM branch earn a place in the ensemble? (variant P2) |
| [`wf_llm_brief_control.md`](wf_llm_brief_control.md) | The shuffled-brief control — is the LLM reading the brief at all? |
| [`wf_llm_logprob_scoring.md`](wf_llm_logprob_scoring.md) | Token logprobs vs a verbalised 0–100 score. Fixed the granularity completely and made ranking **worse**. |
| [`wf_llm_benchset_a.md`](wf_llm_benchset_a.md) | Set A — the run itself, at 2.19% prevalence. |
| [`wf_llm_benchset_a_findings.md`](wf_llm_benchset_a_findings.md) | **The benchmark decision doc.** Zero-shot clears the free cosine baseline on 3 of 8 against a pre-registered ≥6/8; the model ranking scrambles versus TIRI's pools; an induced brief is worth +0.057 ROC-AUC for one $0.006 call — at cold start only. |
| [`wf_llm_benchset_a_summary.md`](wf_llm_benchset_a_summary.md) | Every method and metric side by side, one table. |
| [`wf_llm_benchset_a_baselines.md`](wf_llm_benchset_a_baselines.md) | The cold-start baselines, and the case-control sampling gate that had to pass before any spend. |
| [`wf_llm_benchset_a_low_label.md`](wf_llm_benchset_a_low_label.md) | What 60 labels per collection buy, spent three ways. |
| [`wf_llm_benchset_a_induced_features.md`](wf_llm_benchset_a_induced_features.md) | Does the induced brief help features that do **not** read it? |
| [`wf_llm_benchset_a_ensemble.md`](wf_llm_benchset_a_ensemble.md) | Does the induced brief survive into the per-silo ensemble? **+0.001 — no.** |

## Label budget, coverage, and input pre-checks

| Report | What it is |
|---|---|
| [`wf_label_budget_shape.md`](wf_label_budget_shape.md) · [`_set_b`](wf_label_budget_shape_set_b.md) | How many positives, how many negatives, and what the query costs. Set A, then set B as the check. |
| [`wf_text_input_precheck.md`](wf_text_input_precheck.md) | What text the embedding actually sees — the $0 pre-check for probe P-TX. |
| [`wf_usecase_coverage_cleantech_jasper.md`](wf_usecase_coverage_cleantech_jasper.md) · [`_qwen4b`](wf_usecase_coverage_cleantech_qwen4b.md) | Where new data would diversify the training set, run under two encoders. Read both — an answer that holds on only one is a property of that encoder. |

## Spec quality — the remaining per-surface ablations and the linter

Companion tables to the spec-quality block above, split by surface. The consolidated answer
is [`wf_spec_quality_answer.md`](wf_spec_quality_answer.md); these are its working.

| Report | What it is |
|---|---|
| [`wf_spec_quality_ablation_a_combos.md`](wf_spec_quality_ablation_a_combos.md) · [`_b_combos`](wf_spec_quality_ablation_b_combos.md) · [`_tiri_combos`](wf_spec_quality_ablation_tiri_combos.md) | Field-combination ablations on set A, set B, and TIRI's own six. |
| [`wf_spec_linter_benchset.md`](wf_spec_linter_benchset.md) | The linter run over the 28 benchset specs — **0 findings on all 28**, which is what makes the benchset confirmation unrunnable (see the feasibility report above). |

## Hand-offs

| Report | What it is |
|---|---|
| [`wf_sibling_repo_corrections.md`](wf_sibling_repo_corrections.md) | Five claims in the sibling `academic_agent` repo that this week's measurements changed. A hand-off for that repo's owner; nothing here acts on it. |

## Figures and tables

`reports/` also holds the `.png` / `.csv` / `.json` artifacts the notebooks and scripts
write. They are named after whatever produced them (`wf_ensemble_fold_pipeline_*`,
`eda_quickstart_*`), so each one traces back to its source. Anything carrying a
**`benchset_v1`** prefix belongs to the benchmark corpus, not the six TIRI questions —
see the section above before comparing it with anything else here.
