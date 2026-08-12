# Reports — the decision trail

Twenty write-ups covering what was tried, what was measured, and what was rejected.
They were written over four weeks and **they do not all still hold**, so every one now
opens with a `Status:` line. This index is the same information, sorted.

A note on what the statuses mean — and which critic is speaking. `current` /
`historical` / `superseded` are **navigational judgements made when tidying the repo**,
not measurements. The numbers inside each report are unchanged and remain what they
were when measured; what these labels tell you is whether a later document revised the
*conclusion* drawn from them.

---

## Start here

| Report | What it is |
|---|---|
| [`wf_ensemble_final_recommendations.md`](wf_ensemble_final_recommendations.md) | **The single decision doc.** Every architecture choice, its evidence, and a confidence grade — plus an explicit "do not re-propose these" list. If you read one file in this folder, read this. |

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

## Figures and tables

`reports/` also holds the `.png` / `.csv` / `.json` artifacts the notebooks and scripts
write. They are named after whatever produced them (`wf_ensemble_fold_pipeline_*`,
`eda_quickstart_*`), so each one traces back to its source. Anything carrying a
**`benchset_v1`** prefix belongs to the benchmark corpus, not the six TIRI questions —
see the section above before comparing it with anything else here.
