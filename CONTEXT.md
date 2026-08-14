# CONTEXT.md — what a new agent needs to know before touching this repo

`CLAUDE.md` says how to work here. This file says **what is true here**, and it exists
because several of the facts below are not derivable from the code, the data, or the git
history — they came out of a working session and would otherwise have to be rediscovered
(expensively, and in at least two cases wrongly).

Read this before proposing modelling work. Most bad suggestions in this project are bad
for one of the reasons listed here.

---

## 1. What TIRI is actually for

TIRI is one part of a larger product, not a standalone bootcamp exercise. The product goal:

- A **high-recall (F2) ensemble** that can be fine-tuned for a new use case, run over
  **100k+ papers**, at **low cost**, **100% deterministically**, on **sovereign
  infrastructure**.
- **Precision optimisation comes later.** Users incrementally label and prune toward a
  higher F1. Do not trade recall for precision in this phase.
- **Every new use case arrives with labelled data and a written brief.** How much data,
  and what brief format, are being settled by two separate experiments. Zero-label cold
  start is therefore a *transient*, not the operating regime — but see §4, because at
  realistic prevalence "arrives with labels" is a stronger assumption than it sounds.

### The constraint that decides most architecture questions

**Use cases are per-customer and siloed. There is no shared pool of papers — only shared
tooling.** Each customer's model is trained on their own labels, in their own silo.

Consequences that are easy to get wrong:

- **No pooled/global model ever ships.** What crosses the silo boundary is a *configuration*
  — feature definitions, model class, default hyperparameters, filter rules — never weights
  and never data.
- **Leave-one-use-case-out (LOGO) is a defaults-selection instrument, not a production
  estimate.** It answers "which feature families should we ship to a customer we have no
  labels for". Never quote a LOGO score as "how well TIRI works".
- **Within-silo (grouped k-fold inside one use case) is the production surface.** That is
  the number that means something.
- Hyperparameter search happens **once, centrally, in the LOGO loop**, where six use cases
  of evidence exist. Each silo then just fits weights. Per-customer grid search at a few
  hundred labels would overfit (see `reports/wf_ensemble_report.md` §4).
- Confounds that encode *use-case identity* (source provenance, `has_venue`, `year`) only
  hurt when pooling. **Inside a silo they are admissible** — several warnings in the older
  reports are pooling-specific and dissolve here.

Use cases are uniformly **technology / hard science / industry** — never social science.
That is why a shared non-topical axis (applied-vs-foundational maturity) is plausible at
all, and it is the one paper-only feature family worth testing for portability.

Pools are built from queries derived from the use case and the labelled set, plus citation
harvesting from the labelled set, then narrowed by **deterministic filtering** (not more
queries). Drift, retraining cadence and multi-labeller questions are **deliberately out of
scope right now**: the current target is a versioned model, high recall, a pinned use case,
one labeller.

---

## 2. The central technical finding

**Relevance is a property of the (brief, paper) pair, not of the paper.**

Every feature in this repo before mid-2026 described the paper alone. Measured
consequences, all reproducible:

- A LogisticRegression predicts **`use_case_key` from the paper embedding at 96.2%
  accuracy** (majority class 19%). The embedding is not noisy — it is a use-case
  fingerprint.
- So a model trained across use cases learns *which use case this is*, which is
  definitionally non-transferable. LOGO collapses to ~0.54 ROC-AUC for that reason.
- **This is not "overfitting from too many features."** Ten brief-relative scalars beat a
  384-dimension embedding on LOGO (0.642 vs 0.537). It was never the feature count; it was
  the feature *kind*.
- Strongest form: at zero labels, **unsupervised cosine-to-brief beats every supervised
  cross-domain model** (LOGO WSS@95 0.082 vs 0.042 best supervised; ROC-AUC 0.693 vs 0.610).
  A model trained on other customers is worse than no model at all.
- The corroborating clue was already in the repo: of eleven metadata features tested, the
  only one that worked was term overlap — the only one that read the brief.

**Operating rule this implies:** cosine-to-brief below ~25 in-silo labels, supervised
in-silo model above. Never a model trained on other customers' data.

Things that do **not** fix the transfer problem (both tested, both negative): per-use-case
mean-centring the embedding (0.532 → 0.529), and adding metadata.

---

## 3. Data facts that will bite you

| Fact | Why it matters |
|---|---|
| The live label is **`triage_label`**, not `review_label` (which is empty but for one row). Values: `positive` 1067, `negative` 785, `pass` 543, null 478. | 1,852 usable labelled rows. |
| **`pass` is 84.7% missing-abstract** (vs 5–7% for decided rows). Only 83 `pass` rows have an abstract. | It is a data-completeness artefact, not analyst hesitation. Dropping it is correct and keeps the model binary. Do not build an ordinal target on it. |
| `year` ranges differ wildly by use case (`ner` 2023–26, `tech_forecasting` 2025–26, `solar_leo` 1974–2026). Some rows are dated 2027. | Year is a use-case fingerprint when pooling; fine within a silo. |
| **`from_*` columns are attribution, not capability** — retrieval de-duplicates, so a paper is credited to whichever arm recorded it first. | Any analysis of "which retrieval arm could find this" using these columns is measuring the storage convention. This mistake has already been made once. |
| `terms_exclude` is **empty for 3 of 6** use cases; `performance_criteria`, `decision_*`, `trl_*` are sparse; `constraints_scale` is empty everywhere. | Features over these must emit NaN + an indicator. **NULL is not 0.** Feed the brief-format experiment: `constraints_scale` never got filled. |
| Pools run **26–77% positive**. Production will be low single digits. | Every F2 number and every calibrated threshold in the older reports was measured at ~20x production prevalence. SYNERGY (1.7–14.8%) is the only prevalence-realistic surface available. |
| `scripts/embedding_utils.py:get_use_case_text` claims the export lacks the brief and falls back to the use-case **name**. That was true once; the brief columns exist now. | The `scripts/*.py` path scores against a 2–4 word name. The notebooks build a proper brief. Do not assume they agree. |

---

## 4. Open risks nobody has closed

- **At production prevalence, a random 25-label bootstrap contains almost no positives.**
  Measured on SYNERGY's 1.7% review: **57% of random 25-label draws contain no positive at
  all** and cannot train anything (28% at n=50, 22% at n=100). "Use cases arrive with
  labels" is safe only if those labels were *actively selected* — e.g. by labelling the top
  of a cosine-to-brief ranking — not randomly sampled. Confirm how the labelling project
  sources them.
- ~~**The query-conditioned advantage is contingent on brief format, not automatic.**~~
  **CLOSED — it was a brief-format artefact, not a property of SYNERGY.** The original
  finding stands as measured (shuffled-brief passes on TIRI 5/6, +0.155; fails on old
  SYNERGY 1/3, −0.012), but `benchset_v1`'s re-briefed SYNERGY collections pass the control
  **decisively**: AUC 0.777 → **0.498**, F2@own → **0.000**, predicted-positive rate →
  **0.000** on 8/8 collections (`reports/wf_llm_benchset_a_findings.md` §4). What did *not*
  survive is the diagnosis. "Curated inclusion terminology looks load-bearing" is true for
  the lexical block and **false for an LLM reader**: stripping the term lists back to the
  review's raw abstract leaves ranking identical (0.777 both) and *improves* the operating
  point (recall 0.559 vs 0.484 at equal reading cost). **Term lists help keyword matchers
  and hurt readers** — do not generalise "brief quality matters" into "add term lists".
- **Model ranking is not stable across prevalence regimes.** Qwen3-4B is mid-pack in-repo
  and *last* on SYNERGY; Qwen3-8B wins at realistic prevalence. Recall metrics only
  discriminate where there is room to skip — our 26–77% pools are a poor surface for
  judging a recall-oriented system. **Reproduced a second time, on generative models:** the
  four-model LLM ranking scrambles completely between TIRI and benchset set A —
  `gemma-4-31b` goes 2nd of 4 to *last*, `nemotron-3-super-120b` 3rd to *1st*
  (`wf_llm_benchset_a_findings.md` §2). Treat this as the default expectation, not a
  surprise: **never carry a model choice across a prevalence regime without re-measuring.**
- **Selection-on-holdout.** Many decisions (11 embedding models, combination methods,
  calibration methods, PCA on/off) were made against LOGO scores, so LOGO is no longer
  unbiased. **23 unused SYNERGY reviews** are the only clean surface left — ring-fence them.
- **Label recall is unmeasured.** With one labeller there is no inter-annotator agreement,
  and a labeller's false negative is indistinguishable from a true negative — it silently
  inflates measured recall. Same problem applies to the deterministic filter's discards.
  One fix covers both: sample the negatives (and the filter's rejects) and re-label blind.
- **No causal account of `solar_leo`'s labels.** They track publication year because the
  pool was seeded from citation-ranked canon. Treat it as a corpus defect: never cite it as
  evidence the system finds relevant work.

---

## 5. Statistical discipline this repo requires

Six use cases, 260–360 labelled rows each, per-use-case scores spanning 0.26–0.85. That is
a small, heterogeneous sample and it has already produced at least one decision made on
noise.

- **Measured noise floors:** seed-to-seed sd is ~0.010 on within-silo ROC-AUC and
  **0.015–0.027 on WSS@95**. The entire spread between the three candidate embedding models
  is 0.027 — one sd. Treat any gap below ~0.03 as *not established*.
- Report **per-use-case win counts** alongside means. A mean over six heterogeneous use
  cases can be won by being good at the easy ones.
- Repeat across seeds and report the spread next to every number.
- Never make an architecture decision from a single held-out use case. Several numbers in
  `reports/wf_embedding_bakeoff.md` Rounds 1–2 rest on `tech_forecasting` alone.

---

## 6. The negative-results register

The most valuable asset here. Measured and rejected, so nobody re-runs them:

| Rejected | Evidence |
|---|---|
| The 11-feature metadata punch list (citation velocity, author count, has_venue, is_english, venue cleaning…) | `reports/wf_featureengineering_review.md` — all together worth +0.002 AUC, and they *hurt* out-of-domain |
| TRL keyword estimate, OpenAlex venue quality, author ORCID | earlier notebooks in `notebooks/experiments/` |
| `relevance_score` as a feature or baseline | `reports/wf_ensemble_report.md` §0 — unversioned, moving, not comparable across use cases |
| Prediction-level stacking / learned blending | tied with plain averaging, twice (`wf_ensemble_report.md`, `wf_embedding_bakeoff.md` §5) |
| spaCy over plain regex for term matching | no gain, more cost |
| SPECTER2 | query mode mismatched to paragraph-length briefs |
| Per-use-case mean-centring of embeddings | 0.532 → 0.529 LOGO, no effect |
| **PCA-64 within a silo** | −0.008 to −0.014 WSS on all three models; §6.2's compress-then-concat win was a *transfer* phenomenon and does not carry to production folds |
| Term overlap as an abstract-length proxy (a suspicion, now closed) | length features alone reach 0.550; removing them costs 0.004 |
| **Prompted LLM screening as a replacement for the ensemble** | `wf_llm_pilot_findings.md` — loses on ranking by 0.044, ties on oracle-F2. Also rejected as a third ensemble branch (best +0.019 AUC against a 0.03 floor) |
| **Zero-shot LLM screening as a replacement for the cold-start cosine rung** | `wf_llm_benchset_a_findings.md` §1 — 3 of 8 collections for *every* model from 20B to 397B, against a pre-registered ≥6/8. All four land within 0.030 of each other |
| **Adding LLM-written `terms_*` lists to a brief, for an LLM reader** | Ranking identical (0.777), operating point *worse* (recall 0.484 vs 0.559). Opposite sign to their effect on the lexical block |
| **P4, the per-criterion checklist prompt** | Worse AUC on 3 of 4 models, F2@own collapses to 0.24–0.50 — reasoning scaffolding makes a model demand *all* criteria |
| **Verbalised 0–100 confidence replaced by token logprobs** | Fixed the granularity completely (tie fraction 0.997 → 0.047) and ranking got *worse* (mean AUC −0.035). `wf_llm_logprob_scoring.md` |
| **`nemotron-3-super` on DeepInfra at corpus scale** | 4 rows/min measured (42 h/cell) from rate-limit backoff, despite a healthy 3.9s p50. Throughput is a capability; benchmark it at target scale |
| **The foreign-brief margin as a brief-quality alarm** (`own_brief_auc − best_foreign_brief_auc`) | `wf_foreign_brief_detector.md` — rebuilt in BM25 space, **8 collections flag that the embedding called healthy** against a pre-registered ≤2, and the two margins correlate at rho 0.33. It is a property of one embedding's geometry, not of the text. Its fallback caption (a labelling-cost forecast) also died: rho −0.929 on set A → **−0.464 on set B with the sign flipping to +0.143** under prevalence control, and never embedding-independent even on A (qwen4b −0.571, CI crossing zero). `wf_foreign_brief_validity_setb.md`. **`NUMBERS.md` N33 should be removed** |
| **Criterion checkability / evidence availability as an automatability score** | `wf_checkability_audit.md` — pooled IQR **0.055** against a pre-registered 0.2, and it ranks `tech_forecasting` (no performance criteria, hardest use case here) 3rd of 6 instead of last. Half the metric's own definition never fired: the broad judgement lexicon scored all 34 use cases **identically**. Do not re-derive this without reading §3 of that report |
| **A single spec-quality *score* of any kind** | Three independent attempts, three failures (the two above plus `NUMBERS.md` N23's criteria-populated count, which measures presence not quality). There is a structural reason: the cost of a defect **inverts** between label regimes — prose damage is ~0.17 at 0 labels and ~0.01 fitted, term damage the mirror — so any single number must average regimes that disagree. `wf_spec_quality_answer.md` §5. **Ship a priced linter instead** (`scripts/spec_linter.py`) |

### 6b. Spec quality — what is settled, so it is not re-litigated

Full answer in [`reports/wf_spec_quality_answer.md`](reports/wf_spec_quality_answer.md); the bars were
pre-registered in `wf_spec_quality_plan.md`, which also carries a dated Amendment recording five
premises corrected **before** any probe ran. Measured on **three surfaces** — benchset set A (burned),
set B (clean, and now spent), and **TIRI's own six use cases**, spanning 1.87% to 57.6% prevalence.

- **Three fields carry essentially all the value.** `objective` (+0.151 / +0.263 / +0.100 above chance
  at 0 labels — the largest single-field effect measured anywhere), `terms_nice_to_have` (+0.105 /
  +0.215 / +0.118, and the *largest* effect on TIRI's own corpus), and `terms_must_include` by **count**
  rather than presence.
- **Three fields are decoration to every consumer on all three surfaces.** `problem_statement`,
  `domain_*` and `terms_exclude` measure **0.000 at 0 labels and ≤0.009 fitted**. Keep them for humans;
  **do not make them required**.
- 🔴 **Do not quote set A's term-damage magnitudes.** `keyword_flood` −0.072 → −0.031, `no_must`
  −0.043 → **−0.010** on clean data; set A's means were carried by `leenaars_2020` alone. The
  *cold-start* findings replicate and strengthen; the *after-labels* term findings roughly halve.
- 🔴 **"Fluff is worse than an empty field" is set-A-specific.** 0.473/0.441 on A and TIRI's 0.499/0.491,
  but **0.594/0.569 on set B** where both beat deletion. Any prose beats no prose.
- **Defects compound, they do not rescue.** 6 of 6 prose×term pairs across two surfaces cost more than
  the sum of their parts. A strong objective does not license a lazy term list.
- **A reader is robust to localised spec damage but not to wrong content** — every single-field
  degradation is inside the noise floor while a deranged brief costs **−0.282**. The signal is
  redundant across fields.
- **The one reader-only defect is a prose contradiction**: an objective asserting what `terms_exclude`
  rejects leaves ranking untouched (+0.001) and moves the operating point **+0.120 fraction-read on 8
  of 8 collections**, while a *fitted* matcher sees nothing at all. S-AL's −0.056 AUC is **not**
  reproduced; the effect is real and AUC was the wrong instrument.
- **The completion test that survived everything** is the shuffled-brief control, not any score.

---

## 7. Where the current work lives

| What | Where |
|---|---|
| Query-conditioned lexical features (Tier 1b) + falsification control | `scripts/lexical_features.py`, `scripts/run_tier1b_control.py` |
| Jasper / Qwen3-4B / Qwen3-8B on WSS@95, both fold surfaces | `scripts/run_embedding_recall_comparison.py` |
| SYNERGY validation at realistic prevalence | `scripts/run_synergy_recall_validation.py` |
| Shared fold/metric helpers (use these, do not re-implement) | `scripts/fold_pipeline_utils.py`, `scripts/embedding_utils.py` |
| Full narrative of the above | `reports/wf_query_conditioned_findings.md` |
| Ensemble v1 — per-silo CatBoost + LogisticRegression, feature/embedding ablation, and what was cut from the original proposal | `reports/wf_ensemble_v1_candidate.md`, `reports/wf_ensemble_v1_results.md`, `notebooks/main/09_ensemble_per_silo.ipynb` |
| CatBoost fitting on this machine — **the rule is "never `thread_count=-1`", not "never local"**. Pass an explicit `thread_count` (`catboost_fn` has always taken one; Modal was passing it all along) and local fitting is fine: measured at **15.5s** for 1,660 rows × 4,625 columns, iterations=50/depth=4, and a full 18-cell per-silo grid over set A ran locally in ~2.5h for **$0**. Route to Modal for genuinely heavy jobs, not on principle | `scripts/run_setA_brief_ensemble.py`; the pathology itself is real and documented in `scripts/run_ensemble_candidate.py`'s docstring |
| Ensemble v2 — hyperparameter tuning, nested combiner-weight selection, the Qwen3-8B SYNERGY swap, a 3-lever diversity sweep (SVM/lexical-only/k-NN as a third branch, all rejected, each for a documented reason), and a LOGO-based central hyperparameter search (LogReg `C=1.0` found under-regularizing; not adopted for the 6 shipped use cases but recommended as the starting default for new ones) | `reports/wf_ensemble_v2_experiments.md` (the full running log, §1-16); Modal functions consolidated in `scripts/modal_ensemble_experiments.py` — **do not split Modal functions across files**, see that file's docstring |
| Final, synthesized architecture recommendation — one decision doc pulling together v1 + v2, confidence-graded, with explicit rejects and caveats | `reports/wf_ensemble_final_recommendations.md` |

| LLM screening — the pilot on TIRI (12-cell grid, prompt variants, third-branch blend) and the set-A run at 2.19% prevalence (brief-format ladder, induced rule sets, case-control sampling) | `reports/wf_llm_pilot_findings.md` and `reports/wf_llm_benchset_a_findings.md` are the two decision docs; `scripts/llm_pipeline_utils.py` is the harness, `scripts/benchset_metrics.py` the population-metric layer, `notebooks/main/11_llm_benchset_a.ipynb` the diagnostic |

**🟢 Brief quality is the largest measured lever found so far, and it is not LLM-specific.**
A rule set induced from 60 in-silo labels (`scripts/induce_rule_set.py`, one $0.006 call per
collection) is worth, on identical held-out rows at 2.19% prevalence
(`wf_llm_benchset_a_findings.md` §5, §5b):

| Consumer of the brief | supplied → induced | Δ |
|---|---|---|
| BM25 + term-overlap block, fitted on the same 60 labels | 0.733 → **0.798** | **+0.065** |
| LLM reader, same model and prompt | 0.777 → **0.834** | **+0.057** |
| `bm25_nice` alone, unfitted | 0.605 → **0.696** | **+0.091** |
| **cosine-to-brief** (`qwen4b` / `jasper`) | 0.774 → 0.778 / 0.768 → 0.763 | **+0.004 / −0.005** |

For scale, the entire spread across four model families from 20B to 397B is 0.030 — a better
brief beats a 13× larger model.

**But it is a cold-start lever and nothing else.** Folded into the per-silo CatBoost+LogReg
ensemble (3 seeds, identical feature width) the same swap is worth **+0.001**, with **0 of 8
silos** moving past the noise floor on any branch (`wf_llm_benchset_a_findings.md` §5c). The
whole +0.065 is redundant with the 4,608 embedding dimensions the ensemble already sees; it
only looked like new information because the isolated block could not see them. **Do not ship
the induced brief into the ensemble** — it costs an LLM call per collection and buys nothing.

The ladder, every rung on the same held-out rows at 2.19% prevalence:

| labels per silo | method | mean ROC-AUC |
|---|---|---|
| 0 | cosine-to-brief | 0.774 |
| 60 | BM25 + overlap, supplied brief | 0.733 |
| 60 | BM25 + overlap, **induced** brief | 0.798 |
| 60 | LLM reader, **induced** brief | **0.834** |
| 60 | LogReg on the Qwen3-4B embedding | 0.844 |
| ~80% of the silo | per-silo ensemble | **0.884** |

So §1's rule gains a middle rung: below ~25 labels cosine-to-brief; **at a few dozen labels an
induced brief is worth +0.060 over it for one $0.006 call**; with enough labels to train, train.
The crossover between the last two is unmeasured (60 and ~80%-of-silo are the only points), and
it is the number that decides when to stop paying for a brief.

The split is mechanical and worth remembering: a brief carries **vocabulary** (BM25 and overlap
consume it directly), **instructions** (only a reader acts on them), and **topic** (all a cosine
can see, and the shipped brief already had it right because `objective` is the review's own
abstract — text that looks like the papers). So **the cold-start rung to improve with a better
brief is the lexical block, not the cosine** — which inverts the intuition, because the cosine is
the *stronger* cold-start feature on this corpus (0.774 vs 0.733) and the unimprovable one. With
the induced brief the ordering flips, 0.798 vs 0.778.

Note this does **not** license adding term lists generally: LLM-written `terms_*` are worth
nothing to a reader and cost it recall, while *label-derived* ones are worth +0.091 to a matcher
(§4 above, §5b). Term lists are for matchers, and only if derived from labels.

**The bound on it:** `synergy_moran_2021` is a collection where a 60-label linear probe on the
embedding reaches 0.639 while **no** LLM under any of four briefs — including one distilled
from those exact labels — exceeds chance. Some inclusion rules are learnable in representation
space and **not statable as a rule**. Do not read "distil labels into a brief" as a general
replacement for training.

**The shuffled-brief control is the pattern to copy.** Any feature claiming to read the
brief must be rebuildable against deliberately wrong briefs. If it still scores well, it is
measuring something generic and should be thrown away. `build_lexical_features(df,
brief_map=...)` exists for exactly this reason — keep that seam in anything you add.
