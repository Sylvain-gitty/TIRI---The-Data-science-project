# Do better-written use cases move the baseline model?

Status: **measured.** `scripts/optimise_usecases.py` rewrote TIRI's six specs per [`wf_spec_quality_answer.md`](wf_spec_quality_answer.md); this re-ran `notebooks/main/06_baseline_logreg.ipynb`'s pipeline — same feature list, same splits, same seeds — on each arm. The papers never change, so their 8,704 embedding columns are reused untouched and the only embedding work is 18 short brief strings per arm.

🟢 clears the 0.03 noise floor · 🟡 real but under it · ⚪ engineering finding

## 0. How to read any number here

**The only difference between the arms is the text of six spec files.** Same papers, same paper embeddings, same splits, same seeds, same pipeline. So any difference is attributable to the brief — which is why the *provenance* of the rewrite matters more than the rewrite itself.

🔴 **The rewrite never saw a label, a paper, or the corpus.** Every token in the optimised specs traces to another field of the same spec file — `performance_criteria`, `constraints`, `decision_criteria` and `notes` are all analyst-written and **read by neither** feature path. The optimisation moves the analyst's own words into the fields the model actually reads; it adds no knowledge. Had the briefs been written by reading the relevant papers, the cold-start features match brief text against paper text and the gain would be manufactured — `DATA_BRIEF.md` honest-limit #2 records that failure in the benchset corpus. Run `optimise_usecases.py --provenance` to audit the mapping.

**The three arms.** `baseline` = the six specs as the analysts wrote them. `optimised` = objective enriched *and* term lists topped up. `prose_only` = objective enriched, term lists left exactly as written. The third arm exists because the two halves of the rewrite turn out to pull in opposite directions, and only a separated arm can show that.

**What each metric means, and what it reads if the model does nothing.**

| metric | what it measures | value if the model is useless |
|---|---|---|
| `roc_auc` | ranking quality: the chance a random relevant paper outranks a random irrelevant one | **0.500** (a coin flip) |
| `f2` | the screening-appropriate accuracy score — weights recall 4x precision, because missing a relevant paper costs more than reading an irrelevant one | 0 if it never predicts positive; ~0.73–0.58 if it predicts positive for everything |
| `recall` | of the papers that really were relevant, the share the model flagged | 1.000 if it flags everything — which is why `precision` and `pred_pos_rate` sit beside it |
| `precision` | of the papers the model flagged, the share that really were relevant | **0.576** — the prevalence, i.e. what flagging at random gets you |
| `avg_precision` | area under the precision-recall curve; more informative than AUC when positives are rare | **0.576** (the prevalence) |
| `recall_at_10pct` / `_20pct` | if an analyst read only the top 10% / 20% of the ranked list, what share of the relevant papers would they have found | 0.10 / 0.20 |
| `wss_at_95` | Work Saved over Sampling at 95% recall: the extra share of the pile you can skip while still finding 95% of what matters | **0.000** |

All threshold metrics are at the model's own 0.5 cutoff, nothing tuned.

⚠️ **A gap under 0.03 is not established** (`CONTEXT.md` §5's seed-to-seed noise floor).

⚠️ **This is a POOLED model, and `CONTEXT.md` §1 says no pooled model ever ships** — a LogReg reads `use_case_key` off the raw embedding at 96.2% accuracy, so a pooled gain can be use-case identity rather than brief quality. The per-use-case `test/*` rows are the production-relevant view.

⚠️ **Prevalence is 57.6% positive**, roughly 20x production. Every F2, precision and threshold number here is measured in the wrong regime and does not transfer (`CONTEXT.md` §3).

## 0b. ⚪ Instrument checks — is this the backbone, or a lookalike?

"Same pipeline" has to mean the same pipeline, or part of the difference between arms is the harness. Three checks, run every time:

- **A — cosine and rank code.** Rebuilding `cos_brief_*` and `rank_cos_brief_*` from the cached original brief vectors reproduces the shipped columns to **6.8e-15**.
- **B — lexical code.** Rebuilding all 22 `lex_*` columns reproduces every *value* column exactly (to 7.1e-15), with the NULL pattern unchanged. **2 of 40,656** cells differ, all in `lex_rank_bm25_must` — see the reproducibility note below.
- **C — the brief string itself.** Re-embedding the ORIGINAL six briefs and comparing to the vectors that actually built `papers_fe.parquet` gives cosine **0.999882** at worst — jasper and qwen4b return exactly 1.000000 on all six, and the shortfall is qwen8b, which is served over an API rather than run locally. 1.000000 means this script writes the same brief text notebook 04 did.

🔴 **These checks are not decoration — they caught two real bugs, and the second one was larger than the effect being measured.**

1. **The brief was being formatted wrongly.** The first version of this script embedded `"use_case_name: ...\nproblem_statement: ..."`, whereas `wf_embedding_model_bakeoff.ipynb` cell 6 — the notebook that wrote the cache notebook 04 reads — joins the field *values* with spaces and no field names. Part of the measured "gain" was the formatting change, not the rewrite.
2. **The spec files on disk are not what the corpus carries.** `01_data_compile.ipynb` takes `use_case_name` from its own hand-written registry, not from the JSON's `name` field, and for `solar_leo` those differ in capitalisation: the JSON says *solar cells for low earth orbit satellites*, the corpus says *Solar Cells for Low Earth Orbit Satellites*. Embedding the JSON version instead moved that one brief's vector by cosine 0.991 — **about five times the size of the entire effect this report measures** — and changed 2 of 1,848 `lex_rank_bm25_must` values, which is small enough to be mistaken for a rounding tie and waved through. Both arms are therefore built from the corpus values, with only genuinely-rewritten fields substituted in.

⚪ **Reproducibility note, found while chasing bug 2 and worth recording on its own.** `lex_rank_bm25_must` is **not bit-reproducible across processes**. BM25 adds up one contribution per query term, and the terms come out of a Python set, whose iteration order changes with `PYTHONHASHSEED` — so the sum lands ~1e-15 apart between runs. Two `solar_leo` papers score exactly equal, and that 1e-15 decides which of them pandas' average-tie rank puts first. Measured over four seeds the affected cell count was 2, 2, 0, 2 out of 1,848. It is far too small to touch any number in this report, and it is not a bug in this script — it is a property of the shipped feature pipeline, and it is the reason this check is written on *cell count* rather than on a magnitude tolerance, where a genuine two-cell regression would have hidden underneath it.

**What actually changed in each spec.** Everything not listed is byte-identical to the corpus:

| use case         | optimised                                               | prose_only                        |
|------------------|---------------------------------------------------------|-----------------------------------|
| carbon_capture   | `objective`, `domain_industry`                          | `objective`, `domain_industry`    |
| cement_binders   | `objective`, `terms_must_include`, `terms_nice_to_have` | `objective`                       |
| ner              | `objective`, `terms_must_include`                       | `objective`                       |
| soil_microbiome  | `objective`                                             | `objective`                       |
| solar_leo        | `objective`, `terms_must_include`, `domain_application` | `objective`, `domain_application` |
| tech_forecasting | `objective`                                             | `objective`                       |

Note that `prose_only` is not *purely* the objective: where `domain_industry` or `domain_application` was empty, it is filled from the spec's own `name` or `technology_focus`. That affects two use cases and is part of the cosine brief, so it is named here rather than buried.

**Did the rewrite actually change the features?** Correlation between each arm's feature and the baseline's — 1.0000 would mean the comparison is empty:

- `cos_brief_jasper [optimised]` r=0.9718
- `cos_brief_qwen4b [optimised]` r=0.9748
- `cos_brief_qwen8b [optimised]` r=0.9731
- `lex_bm25_obj [optimised]` r=0.7565
- `lex_bm25_must [optimised]` r=0.8193
- `cos_brief_jasper [prose_only]` r=0.9718
- `cos_brief_qwen4b [prose_only]` r=0.9748
- `cos_brief_qwen8b [prose_only]` r=0.9732
- `lex_bm25_obj [prose_only]` r=0.7565
- `lex_bm25_must [prose_only]` r=1.0000

## 1. Cold start — the regime this was supposed to move

Each brief-reading feature scored **raw**, per use case: no labels, no fitting, no threshold, no split. Every row is held out from a model that does not exist. This is the rung a better spec is supposed to help, and it is the one the fitted baseline in §2 cannot see — §2 is fitted on ~1,478 labels, **24x** the 60-label budget at which `wf_spec_quality_answer.md` §2 found brief quality had already stopped mattering.

| feature               | baseline | optimised | prose_only | d_optimised | d_prose_only | better_optimised | better_prose_only |
|-----------------------|----------|-----------|------------|-------------|--------------|------------------|-------------------|
| cos_brief_jasper      | 0.693    | 0.698     | 0.698      | 0.005       | 0.005        | 3/6              | 3/6               |
| cos_brief_qwen4b      | 0.688    | 0.698     | 0.698      | 0.010       | 0.010        | 4/6              | 4/6               |
| cos_brief_qwen8b      | 0.663    | 0.689     | 0.689      | 0.026       | 0.026        | 5/6              | 5/6               |
| lex_bm25_obj          | 0.574    | 0.577     | 0.577      | 0.003       | 0.003        | 4/6              | 4/6               |
| lex_bm25_must         | 0.640    | 0.630     | 0.640      | -0.010      | 0.000        | 2/6              | 1/6               |
| lex_bm25_nice         | 0.619    | 0.615     | 0.619      | -0.004      | 0.000        | 0/6              | 0/6               |
| lex_overlap_must_frac | 0.595    | 0.606     | 0.595      | 0.011       | 0.000        | 2/6              | 0/6               |

*ELI18: this asks "if you had zero labelled papers and could only sort by how well each paper matches the brief, how good would that sort be?" 0.500 is a coin flip. It is the situation every new project starts in, and the only one where the words in the brief are all the model has. `better_*` counts how many of the six use cases improved.*

### 1a. 🟡 The enriched objective helps, by less than the first run said, and only for two use cases

Averaged over the three embedding models, the enriched objective is worth **+0.014** ROC-AUC across the six use cases, and **2 of 6** clear the 0.03 floor:

| use_case         | baseline_auc | optimised_auc | gain   | models_better | clears_floor |
|------------------|--------------|---------------|--------|---------------|--------------|
| carbon_capture   | 0.759        | 0.804         | 0.045  | 3/3           | 🟢 yes        |
| cement_binders   | 0.835        | 0.835         | -0.000 | 2/3           | —            |
| ner              | 0.749        | 0.760         | 0.011  | 2/3           | —            |
| soil_microbiome  | 0.550        | 0.544         | -0.006 | 1/3           | —            |
| solar_leo        | 0.546        | 0.578         | 0.032  | 3/3           | 🟢 yes        |
| tech_forecasting | 0.647        | 0.648         | 0.001  | 1/3           | —            |

**Hard finding.** The gain is real but **concentrated and model-dependent**. `carbon_capture` (+0.045) and `solar_leo` (+0.032) move; the other four sit inside the floor, and `soil_microbiome` is marginally worse. Which embedding model you ask matters as much as which use case: Qwen3-8B gains +0.026 (5/6 use cases better), Qwen3-4B +0.010, and Jasper only +0.005 — Jasper actually *loses* 0.034 on `ner`. A rewrite that helps one encoder and not another is not yet a property of the brief.

🔴 **This supersedes the first version of this report, which claimed +0.027 on Qwen3-4B across 4 of 6 use cases.** That run embedded the brief in a different format from the one notebook 04 uses (check C above), and roughly two-thirds of the headline gain was the formatting change rather than the rewrite. The corrected figure for that same model is **+0.010, inside the noise floor.** The direction of the conclusion did not change; its size did, by a factor of about three.

### 1b. ⚪ Which half of the rewrite moved it — the objective, not the domain fields

`prose_only` rewrites the `objective` on all six specs and *additionally* fills an empty `domain_industry` / `domain_application` on two — and those two are exactly the two that gain. That is either the explanation or a coincidence over six use cases, so the two edits were re-run separately. No new spec files: the same diff, restricted to a subset of fields.

| use_case         | baseline | objective_only | domain_only | prose_only | d_objective | d_domain | d_both |
|------------------|----------|----------------|-------------|------------|-------------|----------|--------|
| carbon_capture   | 0.759    | 0.796          | 0.775       | 0.804      | 0.037       | 0.016    | 0.045  |
| cement_binders   | 0.835    | 0.835          | 0.835       | 0.835      | 0.000       | 0.000    | 0.000  |
| ner              | 0.749    | 0.760          | 0.749       | 0.760      | 0.011       | 0.000    | 0.011  |
| soil_microbiome  | 0.550    | 0.544          | 0.550       | 0.544      | -0.006      | 0.000    | -0.006 |
| solar_leo        | 0.546    | 0.584          | 0.547       | 0.578      | 0.038       | 0.001    | 0.032  |
| tech_forecasting | 0.647    | 0.648          | 0.647       | 0.648      | 0.001       | 0.000    | 0.001  |

**Hard finding: the objective does essentially all of the work.** Enriching the objective alone is worth **+0.0135** of the **+0.0138** the full rewrite delivers; filling the domain fields is worth **+0.0028**, and all of that sits on one use case (`carbon_capture` +0.016). The coincidence was a coincidence.

🟡 **And on `solar_leo` the domain fill is actively slightly harmful**: the objective alone is worth +0.038, both together only +0.032. Inside the floor, so not established — but it means there is no evidence for "fill in every empty field" as advice, and some against.

### 1c. 🔴 The term-list half of the rewrite buys nothing and risks a lot — ship `prose_only`

`prose_only` and `optimised` differ **only** in whether the analyst's term lists were topped up. Every cosine number above is identical between them — the term lists are not read by the cosine path — so these two rows are the whole story:

| | `lex_bm25_must` | `lex_overlap_must_frac` |
|---|---|---|
| topping the term lists up (`optimised`) | **-0.010** (2/6 better) | **+0.011** (2/6) |
| leaving them alone (`prose_only`) | **0.000** | **0.000** |

**The two term features disagree in sign and both averages are well inside the floor, so on average the change is worth nothing.** The reason to reject it anyway is the spread underneath the average: four of six use cases do not move at all, `solar_leo` — whose must-list had only two terms — gains **+0.046**, and `ner` loses **−0.116** from a single added term. That is a coin-flip payoff attached to a large, one-sided downside. `prose_only` takes the identical cosine gain and declines the bet: **enrich the objective, and leave the analyst's term lists alone.**

#### The term-list rule, v1 → v2, and where it stopped

The first version of the rule padded `terms_must_include` to 8 with whatever `domain_technology_focus` and `performance_criteria[].metric` contained. It cost **−0.028** on `lex_bm25_must`. Three named fixes brought that to **−0.010**:

| fix | derived from | effect |
|---|---|---|
| **Never promote a universal evaluation metric** (`precision`, `recall`, `f1`, `auc`…) | `ner` lost −0.118 with "Precision" in its must-list, matching nearly every NLP paper | `ner` no longer takes Precision/Recall/F1; domain-specific quantities like "conversion efficiency" are still allowed and were worth **+0.046** on `solar_leo` |
| **Top up to a floor, never expand past it** | a spec with enough terms has nothing to gain and everything to dilute | `soil_microbiome` (10) and `tech_forecasting` (6) are now **left alone**, at exactly 0.000 |
| **Never truncate** | capping `soil_microbiome`'s 10 hand-written terms at 8 cost −0.077 | "5–8" is a floor and a quality bar, **never a cap** |

🟡 **One regression survives, and I am stopping rather than fixing it.** `ner` still loses **−0.116** from the single term "News text processing". The diagnosis is dilution rather than genericness: its tokens (`news` 0.16, `text` 0.40, `processing` 0.27 document frequency in that pool) are *not* more common than the existing ones (`entity` 0.78, `named` 0.61), but adding three moderately-common tokens to a five-token query raises scores across ~30–40% of the pool and swamps the rare discriminative ones (`disambiguation` 0.055, `linking` 0.109).

🔴 **That points at a document-frequency filter on candidate terms — and I have not built it, deliberately.** This rule has already been iterated twice while watching the same six-use-case measurement. A third fix aimed at `ner` specifically would be tuning the rule on its own evaluation, which is the selection-on-holdout failure `CONTEXT.md` §4 exists to prevent. The df filter is recorded as a **pre-registered proposal** to test on the 28 benchset briefs, a surface not used for any of this — not applied here.

**So `wf_spec_quality_answer.md` §4's MVP row needs one word changed and one exclusion added:** *at least* 5–8 discriminative phrases, and never promote a metric name into a term list.

## 2. The fitted baseline — every fold, both arms

`validate` is the mean over the five inner folds and `validate_sd` their spread; the five `validate/foldN` rows are underneath so the spread is visible rather than asserted. `train` is in-sample on the training pool and is expected to be the highest row in the table — it is a check that the model fitted, not a result. `test` is the outer holdout, touched once.

| arm        | fold                  | n        | pos_rate | f2    | roc_auc | recall | precision | avg_precision | f1    | pred_pos_rate | recall_at_10pct | recall_at_20pct | wss_at_95 |
|------------|-----------------------|----------|----------|-------|---------|--------|-----------|---------------|-------|---------------|-----------------|-----------------|-----------|
| baseline   | train                 | 1478.000 | 0.577    | 0.841 | 0.906   | 0.838  | 0.852     | 0.925         | 0.845 | 0.567         | 0.169           | 0.336           | 0.230     |
| baseline   | validate              | 295.600  | 0.577    | 0.820 | 0.871   | 0.819  | 0.821     | 0.886         | 0.820 | 0.575         | 0.164           | 0.328           | 0.207     |
| baseline   | validate_sd           | 1.140    | 0.002    | 0.017 | 0.011   | 0.019  | 0.018     | 0.011         | 0.015 | 0.014         | 0.008           | 0.006           | 0.033     |
| baseline   | validate/fold1        | 296.000  | 0.574    | 0.811 | 0.858   | 0.812  | 0.807     | 0.867         | 0.809 | 0.578         | 0.165           | 0.335           | 0.160     |
| baseline   | validate/fold2        | 297.000  | 0.579    | 0.820 | 0.881   | 0.814  | 0.843     | 0.895         | 0.828 | 0.559         | 0.163           | 0.320           | 0.213     |
| baseline   | validate/fold3        | 294.000  | 0.575    | 0.846 | 0.876   | 0.852  | 0.823     | 0.882         | 0.837 | 0.595         | 0.154           | 0.331           | 0.253     |
| baseline   | validate/fold4        | 296.000  | 0.578    | 0.801 | 0.861   | 0.801  | 0.801     | 0.892         | 0.801 | 0.578         | 0.175           | 0.328           | 0.200     |
| baseline   | validate/fold5        | 295.000  | 0.576    | 0.821 | 0.881   | 0.818  | 0.832     | 0.893         | 0.825 | 0.566         | 0.165           | 0.324           | 0.211     |
| baseline   | test                  | 370.000  | 0.576    | 0.790 | 0.867   | 0.779  | 0.838     | 0.873         | 0.808 | 0.535         | 0.160           | 0.324           | 0.180     |
| baseline   | test/carbon_capture   | 58.000   | 0.500    | 0.745 | 0.875   | 0.724  | 0.840     | 0.875         | 0.778 | 0.431         | 0.172           | 0.379           | 0.226     |
| baseline   | test/cement_binders   | 52.000   | 0.654    | 0.843 | 0.935   | 0.824  | 0.933     | 0.958         | 0.875 | 0.577         | 0.147           | 0.294           | 0.258     |
| baseline   | test/ner              | 63.000   | 0.714    | 0.804 | 0.793   | 0.800  | 0.818     | 0.877         | 0.809 | 0.698         | 0.111           | 0.267           | 0.093     |
| baseline   | test/soil_microbiome  | 71.000   | 0.254    | 0.536 | 0.808   | 0.500  | 0.750     | 0.677         | 0.600 | 0.169         | 0.333           | 0.500           | 0.006     |
| baseline   | test/solar_leo        | 72.000   | 0.764    | 0.878 | 0.761   | 0.891  | 0.831     | 0.880         | 0.860 | 0.819         | 0.091           | 0.218           | 0.061     |
| baseline   | test/tech_forecasting | 54.000   | 0.593    | 0.737 | 0.801   | 0.719  | 0.821     | 0.860         | 0.767 | 0.518         | 0.156           | 0.312           | 0.080     |
| optimised  | train                 | 1478.000 | 0.577    | 0.849 | 0.906   | 0.847  | 0.853     | 0.924         | 0.850 | 0.572         | 0.170           | 0.337           | 0.236     |
| optimised  | validate              | 295.600  | 0.577    | 0.822 | 0.871   | 0.822  | 0.823     | 0.884         | 0.822 | 0.576         | 0.166           | 0.324           | 0.208     |
| optimised  | validate_sd           | 1.140    | 0.002    | 0.014 | 0.011   | 0.017  | 0.011     | 0.012         | 0.011 | 0.011         | 0.008           | 0.007           | 0.029     |
| optimised  | validate/fold1        | 296.000  | 0.574    | 0.820 | 0.859   | 0.824  | 0.805     | 0.864         | 0.814 | 0.588         | 0.165           | 0.335           | 0.160     |
| optimised  | validate/fold2        | 297.000  | 0.579    | 0.806 | 0.879   | 0.802  | 0.821     | 0.894         | 0.812 | 0.566         | 0.169           | 0.320           | 0.236     |
| optimised  | validate/fold3        | 294.000  | 0.575    | 0.837 | 0.876   | 0.840  | 0.826     | 0.882         | 0.833 | 0.585         | 0.154           | 0.325           | 0.226     |
| optimised  | validate/fold4        | 296.000  | 0.578    | 0.811 | 0.860   | 0.807  | 0.826     | 0.884         | 0.817 | 0.564         | 0.175           | 0.316           | 0.210     |
| optimised  | validate/fold5        | 295.000  | 0.576    | 0.835 | 0.879   | 0.835  | 0.835     | 0.894         | 0.835 | 0.576         | 0.165           | 0.324           | 0.208     |
| optimised  | test                  | 370.000  | 0.576    | 0.795 | 0.870   | 0.784  | 0.843     | 0.878         | 0.813 | 0.535         | 0.160           | 0.315           | 0.177     |
| optimised  | test/carbon_capture   | 58.000   | 0.500    | 0.745 | 0.876   | 0.724  | 0.840     | 0.882         | 0.778 | 0.431         | 0.207           | 0.379           | 0.191     |
| optimised  | test/cement_binders   | 52.000   | 0.654    | 0.843 | 0.941   | 0.824  | 0.933     | 0.963         | 0.875 | 0.577         | 0.147           | 0.294           | 0.258     |
| optimised  | test/ner              | 63.000   | 0.714    | 0.841 | 0.776   | 0.844  | 0.826     | 0.874         | 0.835 | 0.730         | 0.111           | 0.267           | 0.077     |
| optimised  | test/soil_microbiome  | 71.000   | 0.254    | 0.536 | 0.817   | 0.500  | 0.750     | 0.696         | 0.600 | 0.169         | 0.333           | 0.500           | 0.035     |
| optimised  | test/solar_leo        | 72.000   | 0.764    | 0.866 | 0.775   | 0.873  | 0.842     | 0.888         | 0.857 | 0.792         | 0.109           | 0.218           | 0.047     |
| optimised  | test/tech_forecasting | 54.000   | 0.593    | 0.737 | 0.830   | 0.719  | 0.821     | 0.871         | 0.767 | 0.518         | 0.125           | 0.312           | 0.098     |
| prose_only | train                 | 1478.000 | 0.577    | 0.846 | 0.906   | 0.844  | 0.852     | 0.925         | 0.848 | 0.571         | 0.170           | 0.337           | 0.233     |
| prose_only | validate              | 295.600  | 0.577    | 0.822 | 0.872   | 0.822  | 0.826     | 0.886         | 0.824 | 0.574         | 0.166           | 0.328           | 0.200     |
| prose_only | validate_sd           | 1.140    | 0.002    | 0.021 | 0.009   | 0.026  | 0.018     | 0.011         | 0.016 | 0.018         | 0.008           | 0.007           | 0.028     |
| prose_only | validate/fold1        | 296.000  | 0.574    | 0.811 | 0.862   | 0.812  | 0.807     | 0.870         | 0.809 | 0.578         | 0.165           | 0.335           | 0.160     |
| prose_only | validate/fold2        | 297.000  | 0.579    | 0.822 | 0.880   | 0.814  | 0.854     | 0.895         | 0.833 | 0.552         | 0.169           | 0.326           | 0.223     |
| prose_only | validate/fold3        | 294.000  | 0.575    | 0.856 | 0.875   | 0.864  | 0.825     | 0.879         | 0.844 | 0.602         | 0.154           | 0.320           | 0.229     |
| prose_only | validate/fold4        | 296.000  | 0.578    | 0.799 | 0.863   | 0.795  | 0.814     | 0.894         | 0.805 | 0.564         | 0.175           | 0.333           | 0.193     |
| prose_only | validate/fold5        | 295.000  | 0.576    | 0.825 | 0.879   | 0.824  | 0.828     | 0.890         | 0.826 | 0.573         | 0.165           | 0.324           | 0.198     |
| prose_only | test                  | 370.000  | 0.576    | 0.793 | 0.869   | 0.784  | 0.831     | 0.879         | 0.807 | 0.543         | 0.160           | 0.324           | 0.172     |
| prose_only | test/carbon_capture   | 58.000   | 0.500    | 0.745 | 0.876   | 0.724  | 0.840     | 0.890         | 0.778 | 0.431         | 0.207           | 0.379           | 0.191     |
| prose_only | test/cement_binders   | 52.000   | 0.654    | 0.843 | 0.936   | 0.824  | 0.933     | 0.957         | 0.875 | 0.577         | 0.147           | 0.294           | 0.277     |
| prose_only | test/ner              | 63.000   | 0.714    | 0.822 | 0.790   | 0.822  | 0.822     | 0.882         | 0.822 | 0.714         | 0.111           | 0.267           | 0.077     |
| prose_only | test/soil_microbiome  | 71.000   | 0.254    | 0.529 | 0.806   | 0.500  | 0.692     | 0.667         | 0.581 | 0.183         | 0.333           | 0.500           | 0.020     |
| prose_only | test/solar_leo        | 72.000   | 0.764    | 0.878 | 0.765   | 0.891  | 0.831     | 0.883         | 0.860 | 0.819         | 0.091           | 0.218           | 0.047     |
| prose_only | test/tech_forecasting | 54.000   | 0.593    | 0.733 | 0.824   | 0.719  | 0.793     | 0.877         | 0.754 | 0.537         | 0.156           | 0.312           | 0.098     |

**Test-set ROC-AUC with a 95% bootstrap interval:** baseline 0.867 [0.829, 0.902], optimised 0.870 [0.833, 0.905], prose_only 0.869 [0.833, 0.904]. *ELI18: the interval is where the score would land if you kept re-drawing the 370 holdout papers. Three intervals this wide and this overlapped cannot separate three models.*

## 3. Optimised − baseline, and prose_only − baseline

| fold                  | f2 [optimised] | roc_auc [optimised] | recall [optimised] | precision [optimised] | f2 [prose_only] | roc_auc [prose_only] | recall [prose_only] | precision [prose_only] |
|-----------------------|----------------|---------------------|--------------------|-----------------------|-----------------|----------------------|---------------------|------------------------|
| train                 | 0.008          | -0.000              | 0.009              | 0.001                 | 0.005           | 0.000                | 0.006               | -0.000                 |
| validate              | 0.002          | -0.001              | 0.002              | 0.001                 | 0.003           | 0.001                | 0.002               | 0.004                  |
| validate_sd           | -0.003         | -0.000              | -0.003             | -0.006                | 0.004           | -0.002               | 0.006               | 0.000                  |
| validate/fold1        | 0.009          | 0.001               | 0.012              | -0.002                | 0.000           | 0.004                | 0.000               | 0.000                  |
| validate/fold2        | -0.014         | -0.001              | -0.012             | -0.022                | 0.002           | -0.000               | 0.000               | 0.010                  |
| validate/fold3        | -0.009         | 0.000               | -0.012             | 0.003                 | 0.010           | -0.002               | 0.012               | 0.002                  |
| validate/fold4        | 0.010          | -0.001              | 0.006              | 0.025                 | -0.002          | 0.002                | -0.006              | 0.013                  |
| validate/fold5        | 0.015          | -0.001              | 0.018              | 0.003                 | 0.004           | -0.002               | 0.006               | -0.004                 |
| test                  | 0.005          | 0.003               | 0.005              | 0.005                 | 0.003           | 0.002                | 0.005               | -0.007                 |
| test/carbon_capture   | 0.000          | 0.001               | 0.000              | 0.000                 | 0.000           | 0.001                | 0.000               | 0.000                  |
| test/cement_binders   | 0.000          | 0.006               | 0.000              | 0.000                 | 0.000           | 0.002                | 0.000               | 0.000                  |
| test/ner              | 0.037          | -0.016              | 0.044              | 0.008                 | 0.019           | -0.003               | 0.022               | 0.004                  |
| test/soil_microbiome  | 0.000          | 0.008               | 0.000              | 0.000                 | -0.006          | -0.002               | 0.000               | -0.058                 |
| test/solar_leo        | -0.012         | 0.014               | -0.018             | 0.012                 | 0.000           | 0.003                | 0.000               | 0.000                  |
| test/tech_forecasting | 0.000          | 0.028               | 0.000              | 0.000                 | -0.005          | 0.023                | 0.000               | -0.028                 |

## 4. The answer

**How much does an improved use case improve the baseline model for the six initial use cases? On the fitted baseline, by nothing measurable — and that is the expected result, not a failed experiment.**

Every fold moves less than the fold-to-fold spread of the baseline itself. On the held-out test set, F2 goes 0.790 → 0.795 (optimised) and → 0.793 (prose_only); ROC-AUC 0.867 → 0.870 → 0.869. For scale, the five validation folds of the **unchanged baseline** disagree with each other by ±0.017 on F2 and ±0.011 on ROC-AUC. Every delta in §3 is smaller than that, and smaller again than `CONTEXT.md` §5's 0.03 floor.

**Why that is the expected result.** This model is fitted on ~1,478 labels. `wf_spec_quality_answer.md` §2 measured brief quality mattering enormously at 0 labels (0.15–0.26 ROC-AUC) and already not mattering by 60 (≤0.009). At 1,478 labels — 24x that budget — the labels have long since told the model everything the brief could have, and the brief-derived features are 4 of 8,737 columns. **A well-written brief is worth a lot at label 0 and nothing at label 1,478, and this table is the second case.** It is the wrong instrument for the question, and it is reported because it is the instrument that was asked for; §1 is the right one.

**What actually moves, and by how much.** At zero labels the enriched objective is worth **+0.014** ROC-AUC averaged over three encoders and six use cases, reaching +0.045 on `carbon_capture`, and it clears the noise floor on 2 of 6. Topping up the term lists is worth **nothing on average** and carries a −0.116 tail on one use case, so it should not be done.

**One thing this measurement cannot tell you.** The rewrite only moved the analyst's own words between fields of the same file — deliberately, so nothing leaks. The question it answers is therefore *"how much is currently being wasted by putting good material in fields the model does not read?"*, and the answer for these six specs is: a little, concentrated in the two whose objectives were thinnest. It does **not** measure what a genuinely better-informed brief would be worth, because writing one requires knowledge from outside the file, and there is no leak-free way to get that here.
