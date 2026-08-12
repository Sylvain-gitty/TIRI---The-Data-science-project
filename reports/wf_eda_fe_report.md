# Feature engineering recommendations — EDA decision trail

> **Status: historical.** §4 proposed an 11-feature metadata punch list *from EDA alone* — descriptive reads of means, missingness and correlations. Those features were later built and measured, and mostly rejected: see [`wf_feature_plan.md`](wf_feature_plan.md) for the verdicts and [`wf_featureengineering_review.md`](wf_featureengineering_review.md) for the measurement.

This is the decision trail behind Week-3 feature engineering on
`data/processed/papers_combined.parquet` (2,873 papers, 6 use cases, 50 columns): what
the EDA and feature-viability experiments actually found, what to do about each finding,
and why. Written for anyone picking up this work who wasn't in the room for the
experiments — every recommendation below points at the notebook that earned it, so you
can re-run and check rather than take it on faith.

**Scope note:** this is genuinely Week-3 classifier-prep work, not
another latent-space diagnostic like `scripts/compare_*.py`. Nothing here modifies those
scripts.

**How to read this doc:** each section separates a **hard finding** (a number you can
recompute) from a **judgement call** (an interpretation that needs domain context) — same
"label which critic is speaking" convention as the rest of this repo. If you only read
one thing, read the table in §1.

---

## 1. TL;DR — what to actually do

| Do this | Because |
|---|---|
| **Binarize the target**: predict `positive` vs `negative` only | `pass` is 85%+ "no abstract," not a relevance judgement (§2) |
| **Drop** `domain_industry`, `domain_application`, `domain_technology_focus`, `constraints_scale`, `constraints_cost`, `trl_min`, `trl_max` as model features | All are constant-per-use-case or entirely empty — zero row-level information (§3) |
| **Engineer** `citation_velocity`, per-use-case percentile ranks, missingness indicators, a cleaned `venue`, `author_count` | Raw citation/venue columns are confounded or dirty as-is (§4) |
| **Add** the term-overlap feature (whole-word matching against `terms_must_include`/`nice_to_have`/`exclude`) | Real, tested signal on 3 of 6 use cases (§5.2) |
| **Don't add** TRL-estimate, venue-quality (OpenAlex), or author-ORCID features in their current form | All three tested and came back not-viable or too-weak (§5.1, §5.3, §5.4) |
| **Use `StratifiedGroupKFold`**, stratified on `use_case_key`+label, grouped by first author | Plain `StratifiedKFold` lets the same author's papers leak across train/test (§6) |
| **Hold out one full use case** for a final generalization check, never touched by tuning | In-distribution CV and out-of-domain performance differ by 0.25+ AUC — a huge gap (§6) |

---

## 2. Target and label design

**The finding that changes everything:** `has_abstract` rate by `triage_label` —
**positive 93.3%, negative 95.0%, pass 15.3%**. (`notebooks/experiments/wf_data_enrich.ipynb`)

`pass` isn't mostly a relevance judgement — it's mostly a proxy for "there was no
abstract to triage." If you train a 3-class model, `has_abstract` becomes a near-perfect,
trivial predictor of one class, and the model learns "text is missing," not "this paper
is relevant." Any embedding built from title+abstract is degenerate (title-only) for
~85% of `pass` rows too, for the same reason.

**Recommendation:** treat this as binary classification, `positive` vs `negative`.
`review_label` is unusable regardless (99.97% null — 1 non-null value across 2,873 rows).
The 478 never-triaged rows aren't training data; they're the scoring pool the classifier
is ultimately meant to rank.

*Judgement call:* if a later use case genuinely needs a 3-way decision (route "can't
triage" papers differently from "triaged negative" ones), build that as a **separate**
first-stage gate on `has_abstract` — don't fold it into the relevance classifier's label
space.

---

## 3. Columns to drop as model features

Checked cardinality/fill-rate for every categorical candidate before recommending
anything — three "categorical" columns turned out to be constants in disguise.

| Column(s) | What we found | Verdict |
|---|---|---|
| `domain_industry`, `domain_application`, `domain_technology_focus` | Identical value for every row within a use case — **100% redundant with `use_case_key`** | Drop. One-hotting these just re-weights the same 6-way category twice. |
| `constraints_scale` | **Blank in all 2,873 rows** | Drop. Dead column. |
| `constraints_cost` | Populated for exactly 1 of 6 use cases (`carbon_capture`), one repeated string | Drop as a feature (same redundancy problem — it's really just re-encoding "is this carbon_capture"). |
| `trl_min`, `trl_max` | Broadcast from each use case's search brief — **constant within a use case**, not estimated per paper | Drop as-is. See §5.1 for whether a *real* per-paper version is worth building instead. |
| `language` | 99% `en` (2,843/2,873); 9 other languages, single digits each | Collapse to one `is_non_english` boolean, or drop — near-zero variance either way. |

**How to check this yourself before trusting it (or before adding a new column):**
```python
df.groupby("use_case_key")[candidate_col].nunique()
```
If that's `1` for every use case, the column carries no per-paper information — it's
metadata about the use case, not a feature about the paper.

---

## 4. Columns to engineer, and why the raw versions are misleading

### 4.1 `citation_count` — real signal, pointing the wrong way

Two hard findings, both from `wf_data_enrich.ipynb`:

- **`negative`-labelled papers have *higher* mean citation counts than `positive` ones
  in 5 of 6 use cases** (e.g. solar_leo: negative 460 vs. positive 163; cement_binders:
  negative 122 vs. positive 68). Raw citation count is not a quality/relevance proxy for
  this task — a famous, heavily-cited paper can still be topically off-target for a
  narrow use case.
- **Spearman(age, citation_count) = 0.68 overall** (0.47–0.79 per use case) — older
  papers have simply had more time to accumulate citations. Raw citation count is
  largely standing in for age.

**Do this:** engineer `citation_velocity = citation_count / (paper_age + 1)` instead of
using the raw count. It's less age-confounded, though the negative > positive inversion
survives even after this adjustment (mean velocity 15.9 vs. 11.6) — so it's a real
topical-fit signal, not just an age artifact, and it should be read as such (papers can
be *simultaneously* impactful and off-target).

**Never** `fillna(0)` on `citation_count` — null means "never looked up," 0 means
"looked up, found nothing." They're different facts (`CLAUDE.md`'s NULL ≠ 0 rule) and
conflating them changes the actual meaning of the column. Add a
`citation_count_missing` boolean instead of imputing.

### 4.2 `venue` — clean before you encode

- ~27% missing (both `NaN` and empty string mean "not recorded" — treat as one missing
  state, not two).
- High cardinality: 919 unique values across 2,873 rows.
- **Some long venue strings are dirty data**, not venue names — full citation strings or
  paper titles that landed in the `venue` field (confirmed in the outlier review, e.g.
  `"pan, g., zhou, t., zhao, j., ... (2026). graph neural networks..."`).

**Do this:** add a cheap regex filter before any encoding (a string containing a 4-digit
year in parens plus 2+ commas is probably not a venue — null it out instead of encoding
it as a one-off "rare" category). Then prefer the already-clean `from_arxiv`/
`from_openalex`/etc. boolean columns over raw `venue` text for "what kind of source is
this" — they cover most of the same signal with none of the mess. If you still want
`venue` itself, use a smoothed frequency or target encoding (top-N + "other"), fit only
on the training fold, never a plain one-hot.

Do **not** try to build one combined "metadata completeness" score across
citation/venue/author missingness — they don't move together. `pass` papers actually
have the *lowest* venue-missing rate (21.7%) despite having by far the *highest*
citation-missing rate (16.4% vs. 9.5%/12.9%) and being almost entirely abstract-less.
Keep separate missingness indicators per field and let the model find the interactions.

### 4.3 `authors` → `author_count`

Comma-split, whitespace-normalized parse, same convention used throughout this repo's
EDA. Both `NaN` (26 rows) and empty string (42 rows) mean "not captured" — neither is
"0 authors" (a real paper never has zero authors), so both become missing, not 0.

The resulting frequency table checks out against reality — e.g. Karen Scrivener on 15
papers in `cement_binders`, T. Alan Hatton on 7 in `carbon_capture` — both genuinely
prominent, verifiable researchers in those fields, not parsing noise
(`notebooks/experiments/wf_data_enrich.ipynb`, `notebooks/experiments/author_orcid.ipynb`).

*Judgement call, flagged not solved:* the parsing is naive (no identity disambiguation —
"J. Smith" vs. "John Smith" won't merge, and a single "Last, First" author can get
mis-split into two). Good enough for a frequency feature, not for exact author-identity
work.

### 4.4 Pool the 6 use cases carefully, not naively

Scales differ by field for reasons that have nothing to do with relevance — mean
citation count is 3.4 in `tech_forecasting` vs. 222.1 in `solar_leo`. Pooling raw values
risks the model learning "if solar_leo then predict positive" as a shortcut.

**Do this:** compute per-use-case **percentile ranks** of `citation_count`/
`author_count` in addition to the raw/engineered values, and always keep `use_case_key`
as a feature (or stratify by it — see §6) so the model can tell "high for this field"
apart from "high in absolute terms."

---

## 5. New candidate features — what we actually tested, and the verdict

Four feature ideas were built as small, sample-sized viability notebooks in
`notebooks/experiments/`, plus one follow-up. All four ideas were plausible
going in; two came back negative, one came back weak, one came back genuinely useful
(conditionally). Reporting the negative ones plainly is deliberate — this repo's
convention (see `reports/combined_features_notes.md`) is to keep honest
negative results in the trail, not just the flattering ones.

### 5.1 TRL (Technology Readiness Level) estimate — not viable as a keyword heuristic

**Idea:** `trl_min`/`trl_max` are use-case-level constants (§3) — could we estimate each
*paper's own* TRL from its abstract instead, as a genuine per-row feature?

**What we did:** hand-picked 34 papers across all 6 use cases (not a random sample — a
deliberate spread including "keyword trap" cases, e.g. a lab-scale device titled
"Industrial-Grade"), hand-labelled each with a TRL band as ground truth, then scored the
same sample with a transparent keyword heuristic (`"pilot"/"prototype"` → mid,
`"commercial"/"deployed"` → high, etc.).

**Result:** 18/34 (53%) agreement with hand-labels — barely above the 47% you'd get by
always guessing the sample's majority band. It handled the 3 genuine late-stage papers
correctly (3/3) but was weak everywhere else, especially for NLP/ML papers that don't
use physical-science maturity vocabulary at all.

**Verdict: don't build this as a keyword list.** The per-paper idea itself isn't dead —
it would need one LLM call per abstract, not more keyword tuning. Not attempted here;
flagged as the natural next step if this feature is worth the API cost at 2,873 papers.

*A finding worth knowing on its own merits:* none of the 5 sampled `ner` papers fell
inside that use case's own stated TRL window (6–9) — that corpus is almost entirely
academic algorithm papers, nowhere near the maturity level its search brief asked for.
Worth a conversation with whoever wrote that use case's brief.

### 5.2 Term overlap — real, conditional signal; adopt it

**Idea:** score each paper's title+abstract against its own use case's
`terms_must_include`/`terms_nice_to_have`/`terms_exclude` lists (already in the parquet,
broadcast from `usecase.json`) — does the paper actually contain the words the analyst
said mattered?

**What we did:** whole-word matching (not substring — substring matching on `ner`'s own
term "NER" hit 355/541 papers by matching inside "generic"/"generation"; whole-word
matching corrected this to 229/541), tested against `triage_label` and against the
existing `relevance_score`.

**Result:**

| Use case | ROC-AUC | Signal? |
|---|---|---|
| cement_binders | 0.80 | Yes |
| carbon_capture | 0.71 | Yes (beats `relevance_score`'s own 0.61 here) |
| ner | 0.70 | Yes |
| soil_microbiome | 0.50 | No |
| solar_leo | 0.46 | No |
| tech_forecasting | 0.56 | No |

Correlation with the existing `relevance_score` is only moderate (Pearson r 0.10–0.49) —
this is a mostly independent signal, not a duplicate of one you already have.

**Why 3 of 6 fail:** checked, not assumed. `solar_leo`'s term list is nearly tautological
(2 terms, one of which never matches a single paper). `soil_microbiome` has the
*richest* term list (10 terms) but the weakest signal — richness of the term list
doesn't predict whether this feature will work for a given use case.

**Follow-up tested:** does spaCy (lemma matching, hyphenation-normalized phrase
matching, negation-aware handling of `terms_exclude`) beat plain regex whole-word
matching? Best variant gave +0.01 AUC on 2 of the 3 working use cases and rescued none
of the 3 dead ones — **marginal, not worth taking on the spaCy dependency for.** Plain
whole-word regex captures nearly all the same signal for free.

**Verdict: add the plain whole-word version as a feature**, but per-use-case, not as a
blanket assumption it'll help everywhere — check the same per-use-case breakdown for any
new use case before trusting it there.

### 5.3 Venue quality (external, via OpenAlex) — not viable

**Idea:** replace the dirty free-text `venue` column with an external prestige signal
(citation-impact-style metrics from the free, open OpenAlex API).

**What we did:** looked up the 10 known-clean top venues (arxiv, nature communications,
etc.) plus a few known-dirty ones (as a negative-result check), joined the resulting
`2yr_mean_citedness`/`h_index` back onto the real papers published in those venues.

**Result:** venue quality correlates strongly with the venue's own mean
`citation_count` (r≈0.78) but is essentially **uncorrelated with how analysts actually
triaged those papers** (r≈-0.04) — no better than `citation_count` already is (r≈-0.33
with the label). It mostly restates the axis it was meant to improve on, not
complement it. Combined with the ~27% venue-missing rate and dirty strings returning
zero OpenAlex matches (fails safe, but zero coverage), it isn't worth the pipeline
complexity right now.

**Verdict: don't build this.** If revisited later, the dirty-venue-string cleanup in
§4.2 would need to happen first regardless, since it caps this feature's coverage too.

### 5.4 Author ORCID (career paper count) — too weak, coverage-limited

**Idea:** resolve an author's name to their ORCID iD (via the paper's `doi` against the
open Crossref API, then the public ORCID record) to get a "how prolific/established is
this author" feature.

**Result:** **2 of 10 sampled authors (20%) resolved to an ORCID iD** — and coverage
looked publisher-dependent, not prominence-dependent: well-known, clearly prolific
authors already confirmed in this dataset (Hatton, Zunino) still had no ORCID on their
Crossref record for the sampled paper, while less carefully catalogued sources never
carried one at all. Compounding this with the ~94% DOI-availability ceiling, well under
1 in 5 rows could ever get this feature via this path.

**Verdict: don't build this at 20% coverage.** It also only measures total *career*
output, not relevance to a specific use case, which is a conceptual mismatch worth
weighing even if coverage improves later.

---

## 6. Cross-validation and fold design

**The problem with a plain random or `StratifiedKFold` split here:** some authors appear
on many papers in this dataset (Karen Scrivener × 15). A random split can put the same
author's papers on both sides of train/test, letting the model partially memorize an
author's topic/style instead of learning transferable relevance signal — a leakage
channel that's easy to miss because it doesn't look like a bug.

**What we built and tested** (`notebooks/experiments/wf_fold_pca_test.ipynb`):

- Held **`tech_forecasting`** (the smallest use case, 290 rows, and a distinct
  methodological domain — network/graph methods, not a physical-science topic) out
  entirely, untouched by any fold or by any fitted transform (PCA, scalers, percentile
  ranks) until a single final evaluation.
- On the remaining 5 use cases: `StratifiedGroupKFold`, stratified on a composite
  `use_case_key + label` key (so no fold starves of a rare combination like
  `carbon_capture`/`pass`, n=23), grouped on first-listed author (so no author's papers
  split across folds).
- **Verified, not assumed:** zero first-author overlap across folds, checked
  programmatically via pairwise set intersection.

**What the numbers actually showed** (ROC-AUC, `LogisticRegression` on embeddings as the
probe model):

| Stage | In-distribution | Held-out use case |
|---|---|---|
| Naive baseline, `shuffle=False` (misleading — see below) | 0.601 | — |
| Shuffle-only, still no grouping (isolates the artifact) | 0.815 | — |
| **Grouped + stratified (the recommended scheme)** | 0.802 | 0.530 |
| Shuffled-label sanity check | 0.526 (≈ chance, as it should be) | — |

**A methodological trap worth knowing about explicitly:** the first naive-vs-grouped
comparison looked like grouping *helped* (+0.20 AUC) — but that was mostly an artifact of
`embedding_utils.cross_validated_roc_auc` using `shuffle=False` on data stored in
contiguous per-use-case blocks, not a real effect of grouping. This was only caught by
adding the shuffle-only intermediate row above. **Lesson: always isolate one variable at
a time when comparing CV schemes** — don't attribute a delta to the change you made
without checking whether something else changed too.

**The honest cost of doing this correctly:** grouping by author costs ~0.013 AUC
in-distribution versus the shuffle-only number (0.815 → 0.802). That's the real price of
closing a leakage channel — not a free win, and not a reason to skip it.

**The generalization gap is the headline finding of this whole section:** every model
tested — plain embedding baseline, gradient boosting on raw embeddings+metadata, PCA'd
embeddings + logistic regression, and a stacked ensemble of the two — landed at
**0.53–0.59 ROC-AUC on the fully held-out use case**, regardless of model complexity.
Stacking (0.833) essentially matched simple averaging (0.834) in-distribution and gave no
edge out-of-domain either (both 0.586). **A model that looks strong in-distribution
(0.80+) should not be assumed to work on a genuinely new use case** — budget for a
per-use-case calibration or retraining step, don't expect one classifier to generalize
cold to a new research question.

**Recommendation for any future modelling notebook:** reuse this same fold scheme
(composite stratification + author grouping + one use case held out) rather than
building a new one — it's already validated against a shuffled-label sanity check and a
programmatic leakage check, both cheap to rerun if you change the data.

---

## 7. Encoding cheat sheet

| Column | Cardinality/shape | Encoding |
|---|---|---|
| `use_case_key` | 6, unordered | One-hot, or native categorical support (LightGBM/CatBoost) if available |
| `domain_*`, `constraints_*`, `trl_min`/`trl_max` | Constant-per-group or empty | **Don't encode — drop (§3)** |
| `language` | 99% one value | Collapse to `is_non_english` boolean, or drop |
| `venue` | 919 unique / 2,873 rows, ~27% missing, some dirty | Never one-hot. Prefer `from_*` booleans; if kept, smoothed frequency/target encoding, fit on train only |
| `authors` | ~unique per row | Never one-hot — derived features only (`author_count`, repeat-author frequency) |
| A future *real* per-paper TRL (§5.1) | Ordinal, 1–9 | Numeric/ordinal, never one-hot (one-hot would deny that TRL 6 is "more" than TRL 3) |
| `from_arxiv`, `from_openalex`, etc. | Already boolean | Use as-is; consider adding `n_sources` (sum of flags) |

**General rule:** verify a column actually varies at the *row* level
(`df.groupby("use_case_key")[col].nunique()` should be >1 for at least one use case)
before spending any encoding effort on it — §3 exists because three columns here looked
like real per-paper categoricals and weren't.

---

## 8. Dimensionality reduction (PCA / SVD)

The 384-dim embeddings are one consistent model
(`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) across all 6 use cases —
confirmed via `embed_model`/`embed_dim` being constant throughout the parquet, so there's
no mixed-embedding-space problem to solve first.

- **Not required** for tree-based learners (gradient boosting, random forest) — they
  split-find fine on 384 dims directly, and `wf_fold_pca_test.ipynb`'s
  `HistGradientBoostingClassifier` on raw embeddings performed comparably to the
  PCA-reduced linear model (0.832 vs. 0.808 in-distribution).
- **Useful for a linear/logistic model in a stacked ensemble** — reduces
  overfitting/multicollinearity risk from 384 correlated raw dimensions. Tested K ∈
  {10, 30, 50, 100, 200}; **K=30** was the peak of the sweep in
  `wf_fold_pca_test.ipynb` — smallest value tested that was also the best, not a
  compromise.
- **Fit only on the training fold** (or the 5-use-case dev pool, never the held-out use
  case) — the same discipline as every other fitted transform in this pipeline.
  `TruncatedSVD` and `PCA` are equivalent for these already-dense embeddings; `SVD`
  matters more if you ever add a sparse TF-IDF-style feature alongside them.

---

## 9. Checks to run before trusting a model built on this data

1. **Zero-variance-within-use-case scan** for every candidate feature (§3/§7) — cheap,
   catches dead/redundant columns immediately.
2. **Shuffled-label sanity check** — CV AUC should collapse to ≈0.5 on permuted labels.
   Confirmed at 0.526 in `wf_fold_pca_test.ipynb`; rerun this for any new model.
3. **Group-integrity check** — programmatically verify zero author overlap across CV
   folds, don't just assume `StratifiedGroupKFold` did what you expect.
4. **Leave-one-use-case-out generalization check** (§6) — an in-distribution CV score is
   not evidence the model works on a genuinely new research question.
5. **Feature-importance audit for `has_abstract`/raw `citation_count` dominance** — if
   either tops the importance list, that's very likely the §2/§4.1 confound resurfacing,
   not real relevance signal.
6. **Calibration, not just discrimination** — since the eventual output is a ranked
   probability (same pattern as the parked `future_work/train_baseline_classifier.py`'s
   scored-pool design), check reliability curves per use case, not only ROC-AUC.
7. **Cross-use-case near-duplicate check** — `explore_use_cases.ipynb`'s fuzzy
   title-match logic already found near-duplicates; make sure the same (or a near-same)
   paper appearing in two use cases doesn't land in train for one and test for the
   other.

---

## 10. Where everything lives

| What | Where |
|---|---|
| The combined, cleaned dataset | `data/processed/papers_combined.parquet` (see `data/processed/README.md` for the full data card) |
| Descriptive EDA (all the §2/§4 findings) | `notebooks/experiments/wf_data_enrich.ipynb` |
| Feature viability experiments (§5) | `notebooks/experiments/` — `trl_estimate.ipynb`, `terms_overlap.ipynb`, `terms_overlap_spacy.ipynb`, `venue_quality.ipynb`, `author_orcid.ipynb` |
| Fold design + stacked PCA ensemble (§6, §8) | `notebooks/experiments/wf_fold_pca_test.ipynb` |
| Diagnostic-only embedding/NER comparisons (separate scope — see `reports/model_shortlist.md`) | `scripts/compare_embeddings.py`, `compare_ner_models.py`, `compare_combined_features.py`, and their `reports/*_notes.md` write-ups |
