# Do better-written use cases move the baseline model?

Status: **measured.** `scripts/optimise_usecases.py` rewrote TIRI's six specs per [`wf_spec_quality_answer.md`](wf_spec_quality_answer.md); this re-ran `notebooks/main/06_baseline_logreg.ipynb`'s pipeline on the result. $0 beyond 12 short brief embeddings — the papers never change, so their vectors are reused untouched.

🟢 clears the 0.03 noise floor · 🟡 real but under it · ⚪ engineering finding

## 0. How to read any number here

**The only difference between the two arms is the text of six spec files.** Same papers, same paper embeddings, same splits, same seeds, same pipeline. So any difference is attributable to the brief — which is exactly why the *provenance* of the rewrite matters more than the rewrite itself.

🔴 **The rewrite never saw a label, a paper, or the corpus.** Every token in the optimised specs traces to another field of the same spec file — `performance_criteria`, `constraints`, `decision_criteria` and `notes` are all analyst-written and **read by neither** feature path. The optimisation moves the analyst's own words into the fields the model actually reads; it adds no knowledge. Had the briefs been written by reading the relevant papers, the cold-start features match brief text against paper text and the gain would be manufactured — `DATA_BRIEF.md` honest-limit #2 records that failure in the benchset corpus. Run `optimise_usecases.py --provenance` to audit the mapping.

**Metrics.** `f2` weights recall 4x precision — the screening-appropriate one, since missing a relevant paper costs more than reading an irrelevant one. `roc_auc` is ranking quality (0.500 = coin flip). `avg_precision` is the precision-recall area, which is more informative than AUC when positives are rare. All at the model's own 0.5 threshold, nothing tuned.

⚠️ **A gap under 0.03 is not established** (`CONTEXT.md` §5).

⚠️ **This is a POOLED model, and `CONTEXT.md` §1 says no pooled model ever ships** — a LogReg reads `use_case_key` off the raw embedding at 96.2% accuracy, so a pooled gain can be use-case identity rather than brief quality. The per-use-case `test/*` rows are the production-relevant view.

⚠️ **Prevalence is 57.6% positive**, roughly 20x production. Every F2, precision and threshold number here is measured in the wrong regime and does not transfer (`CONTEXT.md` §3).

**Instrument check.** Correlation between the two arms' brief-dependent features — 1.000 would mean the rewrite changed nothing and the comparison is empty:

- `cos_brief_qwen4b [optimised]` r=0.9526
- `lex_bm25_obj [optimised]` r=0.7565
- `lex_bm25_must [optimised]` r=0.8193
- `cos_brief_qwen4b [prose_only]` r=0.9526
- `lex_bm25_obj [prose_only]` r=0.7565
- `lex_bm25_must [prose_only]` r=1.0000

## 1. 🟢 Cold start — the regime this was supposed to move

Each brief-reading feature scored **raw**, per use case: no labels, no fitting, no threshold. This is the rung a better spec is supposed to help, and it is the one the fitted baseline in §2 cannot see.

| feature               | baseline | optimised | prose_only | d_optimised | d_prose_only | better_optimised | better_prose_only |
|-----------------------|----------|-----------|------------|-------------|--------------|------------------|-------------------|
| cos_brief_jasper      | 0.693    | 0.698     | 0.698      | 0.005       | 0.005        | 3/6              | 3/6               |
| cos_brief_qwen4b      | 0.688    | 0.715     | 0.715      | 0.027       | 0.027        | 4/6              | 4/6               |
| lex_bm25_must         | 0.640    | 0.630     | 0.640      | -0.010      | 0.000        | 2/6              | 1/6               |
| lex_bm25_nice         | 0.619    | 0.615     | 0.619      | -0.004      | 0.000        | 0/6              | 0/6               |
| lex_bm25_obj          | 0.574    | 0.577     | 0.577      | 0.003       | 0.003        | 4/6              | 4/6               |
| lex_overlap_must_frac | 0.595    | 0.606     | 0.595      | 0.011       | 0.000        | 2/6              | 0/6               |

**Hard finding — the two halves of the rewrite pull in opposite directions, and the `prose_only` arm separates them cleanly.** Enriching the objective from the analyst's own unread fields lifts the qwen4b cosine by **+0.027 on average and 4 of 6 use cases, with all four gains clearing the 0.03 floor** (`solar_leo` +0.061, `ner` +0.047, `carbon_capture` +0.036, `soil_microbiome` +0.034). Padding the term lists costs **−0.028** on `lex_bm25_must`. **`prose_only` keeps the entire cosine gain and none of the loss** — every term-derived feature returns to exactly 0.000.

### The term-list rule, v1 → v2, and where it stopped

The first version of the rule padded `terms_must_include` to 8 with whatever `domain_technology_focus` and `performance_criteria[].metric` contained. It cost **−0.028** on `lex_bm25_must`. Three named fixes brought that to **−0.010**:

| fix | derived from | effect |
|---|---|---|
| **Never promote a universal evaluation metric** (`precision`, `recall`, `f1`, `auc`…) | `ner` lost −0.118 with "Precision" in its must-list, matching nearly every NLP paper | `ner` no longer takes Precision/Recall/F1; domain-specific quantities like "conversion efficiency" are still allowed and were worth **+0.046** on `solar_leo` |
| **Top up to a floor, never expand past it** | a spec with enough terms has nothing to gain and everything to dilute | `soil_microbiome` (10) and `tech_forecasting` (6) are now **left alone**, at exactly 0.000 |
| **Never truncate** | capping `soil_microbiome`'s 10 hand-written terms at 8 cost −0.077 | "5–8" is a floor and a quality bar, **never a cap** |

🟡 **One regression survives, and I am stopping rather than fixing it.** `ner` still loses **−0.116** from the single term "News text processing". The diagnosis is dilution rather than genericness: its tokens (`news` 0.16, `text` 0.40, `processing` 0.27 document frequency in that pool) are *not* more common than the existing ones (`entity` 0.78, `named` 0.61), but adding three moderately-common tokens to a five-token query raises scores across ~30–40% of the pool and swamps the rare discriminative ones (`disambiguation` 0.055, `linking` 0.109).

🔴 **That points at a document-frequency filter on candidate terms — and I have not built it, deliberately.** This rule has now been iterated twice while watching the same six-use-case measurement. A third fix aimed at `ner` specifically would be tuning the rule on its own evaluation, which is the selection-on-holdout failure `CONTEXT.md` §4 exists to prevent. The df filter is recorded as a **pre-registered proposal** to test on the 28 benchset briefs, a surface not used for any of this — not applied here.

**So the recommended configuration is `prose_only`**: it captures the entire measured gain (+0.027 cosine) with every term-derived feature at exactly 0.000. On this evidence, enrich the objective and **leave the analyst's term lists alone**.

🔴 **The original framing of this section is kept below, because it was my mistake and the record should show it.** `wf_spec_quality_answer.md` §4 asks for "5–8 **precise, discriminative** phrases"; the rule I wrote padded to 8 with whatever `domain_technology_focus` and `performance_criteria[].metric` happened to contain. That is `keyword_flood` with domain words instead of generic ones, and it reproduced that failure mode almost exactly (−0.028 here against −0.031 measured on set B). Two specific mechanisms:

1. **Metric names are terrible must-include terms.** `ner` lost **−0.118** because "Precision" was promoted from `performance_criteria` into `terms_must_include`, where it matches virtually every NLP paper ever written.
2. **"5–8" must be a floor and a quality bar, never a cap.** `soil_microbiome` lost **−0.077** because its 10 hand-written, precise terms were **truncated to 8** to satisfy a recommendation derived from a corpus whose specs happened to carry 6–8.

**So §4's MVP row needs one word changed and one exclusion added:** *at least* 5–8 discriminative phrases, and never promote a metric name into a term list.

*ELI18: this asks "if you had zero labelled papers and could only sort by how well each paper matches the brief, how good would that sort be?" 0.500 is a coin flip. It is the situation every new project starts in, and it is the only situation where the words in the brief are all the model has.*

## 2. Both arms, every fold — the fitted baseline

| arm        | fold                  | n    | pos_rate | roc_auc | avg_precision | f2    | f1    | recall | precision | pred_pos_rate |
|------------|-----------------------|------|----------|---------|---------------|-------|-------|--------|-----------|---------------|
| baseline   | train                 | 1478 | 0.577    | 0.902   | 0.921         | 0.838 | 0.843 | 0.835  | 0.852     | 0.565         |
| baseline   | validate              | 1478 | 0.577    | 0.868   | 0.876         | 0.818 | 0.818 | 0.817  | 0.820     | 0.574         |
| baseline   | test                  | 370  | 0.576    | 0.865   | 0.869         | 0.790 | 0.815 | 0.775  | 0.859     | 0.519         |
| baseline   | test/carbon_capture   | 58   | 0.500    | 0.875   | 0.870         | 0.745 | 0.778 | 0.724  | 0.840     | 0.431         |
| baseline   | test/cement_binders   | 52   | 0.654    | 0.941   | 0.958         | 0.893 | 0.909 | 0.882  | 0.938     | 0.615         |
| baseline   | test/ner              | 63   | 0.714    | 0.742   | 0.874         | 0.788 | 0.805 | 0.778  | 0.833     | 0.667         |
| baseline   | test/soil_microbiome  | 71   | 0.254    | 0.816   | 0.702         | 0.488 | 0.571 | 0.444  | 0.800     | 0.141         |
| baseline   | test/solar_leo        | 72   | 0.764    | 0.746   | 0.852         | 0.870 | 0.865 | 0.873  | 0.857     | 0.778         |
| baseline   | test/tech_forecasting | 54   | 0.593    | 0.818   | 0.881         | 0.742 | 0.780 | 0.719  | 0.852     | 0.500         |
| optimised  | train                 | 1478 | 0.577    | 0.902   | 0.921         | 0.837 | 0.840 | 0.835  | 0.845     | 0.569         |
| optimised  | validate              | 1478 | 0.577    | 0.869   | 0.877         | 0.822 | 0.820 | 0.823  | 0.818     | 0.580         |
| optimised  | test                  | 370  | 0.576    | 0.869   | 0.874         | 0.796 | 0.824 | 0.779  | 0.874     | 0.513         |
| optimised  | test/carbon_capture   | 58   | 0.500    | 0.876   | 0.878         | 0.775 | 0.800 | 0.759  | 0.846     | 0.448         |
| optimised  | test/cement_binders   | 52   | 0.654    | 0.944   | 0.962         | 0.874 | 0.906 | 0.853  | 0.967     | 0.577         |
| optimised  | test/ner              | 63   | 0.714    | 0.737   | 0.869         | 0.804 | 0.809 | 0.800  | 0.818     | 0.698         |
| optimised  | test/soil_microbiome  | 71   | 0.254    | 0.821   | 0.724         | 0.494 | 0.593 | 0.444  | 0.889     | 0.127         |
| optimised  | test/solar_leo        | 72   | 0.764    | 0.753   | 0.858         | 0.873 | 0.873 | 0.873  | 0.873     | 0.764         |
| optimised  | test/tech_forecasting | 54   | 0.593    | 0.857   | 0.900         | 0.747 | 0.793 | 0.719  | 0.885     | 0.481         |
| prose_only | train                 | 1478 | 0.577    | 0.902   | 0.923         | 0.834 | 0.837 | 0.832  | 0.842     | 0.570         |
| prose_only | validate              | 1478 | 0.577    | 0.869   | 0.879         | 0.815 | 0.817 | 0.815  | 0.818     | 0.574         |
| prose_only | test                  | 370  | 0.576    | 0.867   | 0.873         | 0.796 | 0.822 | 0.779  | 0.869     | 0.516         |
| prose_only | test/carbon_capture   | 58   | 0.500    | 0.879   | 0.885         | 0.745 | 0.778 | 0.724  | 0.840     | 0.431         |
| prose_only | test/cement_binders   | 52   | 0.654    | 0.940   | 0.956         | 0.874 | 0.906 | 0.853  | 0.967     | 0.577         |
| prose_only | test/ner              | 63   | 0.714    | 0.749   | 0.875         | 0.804 | 0.809 | 0.800  | 0.818     | 0.698         |
| prose_only | test/soil_microbiome  | 71   | 0.254    | 0.813   | 0.705         | 0.494 | 0.593 | 0.444  | 0.889     | 0.127         |
| prose_only | test/solar_leo        | 72   | 0.764    | 0.747   | 0.855         | 0.884 | 0.875 | 0.891  | 0.860     | 0.792         |
| prose_only | test/tech_forecasting | 54   | 0.593    | 0.837   | 0.891         | 0.747 | 0.793 | 0.719  | 0.885     | 0.481         |

## 2. Optimised − baseline

| fold                  | f2 [optimised] | roc_auc [optimised] | recall [optimised] | f2 [prose_only] | roc_auc [prose_only] | recall [prose_only] |
|-----------------------|----------------|---------------------|--------------------|-----------------|----------------------|---------------------|
| train                 | -0.001         | 0.000               | 0.000              | -0.004          | 0.001                | -0.002              |
| validate              | 0.004          | 0.000               | 0.006              | -0.002          | 0.001                | -0.002              |
| test                  | 0.006          | 0.005               | 0.005              | 0.006           | 0.003                | 0.005               |
| test/carbon_capture   | 0.030          | 0.001               | 0.035              | 0.000           | 0.004                | 0.000               |
| test/cement_binders   | -0.019         | 0.003               | -0.029             | -0.019          | -0.002               | -0.029              |
| test/ner              | 0.015          | -0.005              | 0.022              | 0.015           | 0.007                | 0.022               |
| test/soil_microbiome  | 0.006          | 0.005               | 0.000              | 0.006           | -0.002               | 0.000               |
| test/solar_leo        | 0.003          | 0.007               | 0.000              | 0.015           | 0.001                | 0.018               |
| test/tech_forecasting | 0.005          | 0.038               | 0.000              | 0.005           | 0.018                | 0.000               |
