# Feature engineering plan — the 11-feature punch list, after measurement

> **Status: current.** The per-feature verdicts on the 11-feature punch list that [`wf_eda_fe_report.md`](wf_eda_fe_report.md) §4 proposed from EDA alone. This is the result; that report is the hypothesis.

**Status:** decision document. Every number here is reproducible from
`notebooks/experiments/wf_feature_validation.ipynb`, which builds all 11 proposed
features and measures them.

**Relationship to `wf_eda_fe_report.md`:** that report's §4 recommended this punch list
from EDA — descriptive reads of means, missing-value rates and correlations. This document
is what happened when the recommendations were actually measured against the labels. Two
of them do not survive their own justification, and this supersedes §4 where they differ.

---

## 1. TL;DR

| | Feature | Decision |
|---|---|---|
| ✅ | `term_overlap_positive` / `_negative` (+ the two regex term lists) | **Build.** The only proposed features with a measured effect. |
| ✅ | `per_use_case_percentile_*` | **Build, pooled stage only.** Provably inert per use case. |
| ⚠️ | `citation_count` + `paper_age` as two columns | **Optional.** Prefer over `citation_velocity` if you want citations at all. |
| ❌ | `citation_velocity` | Drop — does not fix what it was introduced to fix. |
| ❌ | `author_count` | Drop from pooled; reconsider only for wide-year-span corpora. |
| ❌ | `has_venue` | Drop — it measures which API answered, not the paper. |
| ❌ | `venue_is_arxiv_only` | Drop — φ=0.83 with the existing `from_arxiv` column. |
| ❌ | `is_english` | Drop — 98.96% one value. |
| ❌ | venue "long string" cleanup | Don't build — 2 dirty strings vs 40 real conference names. |

**The headline number.** With `tech_forecasting` held out and honest out-of-fold
predictions:

| feature set | OOF ROC-AUC (LogReg) | OOF ROC-AUC (HistGB) |
|---|---|---|
| embedding + `relevance_score` | 0.783 | 0.834 |
| **+ all 8 metadata columns** | **0.782** | **0.836** |

All the metadata engineering together is worth **−0.001 / +0.002 AUC**. That is the
central finding, and it is why this plan is mostly a list of things not to build.

**The conclusion does not depend on `relevance_score`**, which `reports/wf_ensemble_report.md`
§0 has since disqualified as an ensemble feature. Re-run with it removed entirely, metadata
is not merely useless but mildly harmful to transfer: `PCA32 + term_overlap` scores 0.820
OOF / **0.714** on the held-out use case; adding the 5 metadata columns moves that to 0.828
OOF / **0.666** — dev score up, unseen-use-case score down by 0.048. One holdout of 264
rows, so read it as directional, but it is the direction the provenance confound in §2.3
predicts. See `reports/wf_featureengineering_review.md` §6.3 for the full table.

---

## 2. What changed against the original table, and why

### 2.1 `citation_velocity` — the fix doesn't fix it

`citation_velocity = citation_count / paper_age` was introduced because raw
`citation_count` is age-confounded (Spearman(age, citations) = 0.68). The logic was sound.
The result is not: dividing by age moves full-population ROC-AUC by ±0.02 and leaves
**five of six use cases below 0.5** — more citations still means *less* likely relevant.

| use case | `citation_count` | `citation_velocity` |
|---|---|---|
| carbon_capture | 0.471 | 0.466 |
| cement_binders | 0.483 | 0.468 |
| ner | 0.513 | 0.497 |
| soil_microbiome | 0.470 | 0.485 |
| solar_leo | 0.535 | **0.578** |
| tech_forecasting | 0.464 | 0.462 |

Only `solar_leo` clears chance by more than 0.05, and only there is it significant
(Mann-Whitney p=0.031, uncorrected across 30 tests).

The EDA read that motivated it ("mean velocity 15.9 for negatives vs 11.6 for positives")
was **driven by outliers** — one paper in this corpus has 26,506 citations. A rank-based
measure shows the effect is not there.

**Recommendation:** if you want citation information at all, pass `citation_count` and
`paper_age` as **two separate columns** and let a tree model find whatever interaction
exists. A hand-built ratio bakes in an assumption (impact is linear in 1/age) that the
data does not support, and it destroys information the model could have used.

**Watch the denominator either way.** `tech_forecasting` spans 2025–2026 and
`named-entity-recognition` 2023–2026; 642 rows are dated 2026 and one is dated 2027. Any
`age` denominator is near-zero or negative for a large fraction of rows. The notebook uses
`clip(lower=0.5)` rather than the `+1` in the original spec, because `+1` silently turns
"published this year" and "published last year" into a 2× velocity difference that is an
artefact of the constant, not of the papers.

### 2.2 `has_venue` — it measures the API, not the paper

This is the most important finding in the notebook, and it generalises beyond `has_venue`.

Nullness in this dataset is **perfectly determined by which search backend found the
paper** (single-source rows, so nothing is double-counted):

| source | n | `citation_count` NULL % | `venue` NULL % |
|---|---|---|---|
| arxiv | 135 | **100.0** | 0.0 |
| core | 66 | **100.0** | **100.0** |
| crossref | 977 | 0.0 | 22.3 |
| europe_pmc | 404 | 0.0 | **100.0** |
| openalex | 895 | 0.0 | 8.5 |
| pubmed | 180 | **100.0** | 0.0 |
| semantic_scholar | 33 | 0.0 | 3.0 |
| seed | 30 | 0.0 | 3.3 |

These are not noisy correlations — they are the APIs' response schemas. Europe PMC does
not return a venue field; PubMed and arXiv do not return citation counts.

So `has_venue` is a re-encoding of `sources`, which is **already in the parquet** as
`from_*` booleans. And it behaves exactly as a confound should: its **direction flips by
use case** — 0.570 on `carbon_capture` (venue present → positive) and 0.412 on `solar_leo`
(venue present → negative), both individually significant (p=0.012, p=0.002), washing out
to 0.514 pooled.

**Recommendation:** drop it. If you want retrieval provenance in the model, use the
`from_*` booleans that already exist and name them honestly as provenance, so nobody later
reads a venue coefficient as a statement about publication quality.

### 2.3 The trap this opens up

`citation_count_missing` — a feature **nobody proposed** — scores **0.238** on
`cement_binders`. That is further from chance (0.262) than anything in the table except
`relevance_score`. It is a strong, exploitable, statistically real signal.

It is also **entirely an artefact of the search pipeline**: on that corpus, "was found via
pubmed/arxiv/core" predicts *negative*. It says nothing about whether a paper answers the
research question. A pooled model will find it, and it is the single most likely reason a
model that looks good in cross-validation collapses on a new use case whose queries hit a
different mix of APIs.

**Recommendation:** do not add it. Add it to the held-out-use-case checks in
`wf_fold_pca_test.ipynb` as a **negative control** — if a model's performance depends on
it, that model has learned the retrieval pipeline.

### 2.4 `venue_is_arxiv_only` — the definition needs repair, and then still fails

The "TBD" was well placed; the obvious definition is wrong in two directions.

- **41 arXiv papers carry a real journal venue.** arXiv metadata includes `journal-ref`
  once a preprint is published, so `from_arxiv` alone mislabels 41 published papers as
  preprint-only.
- **49 papers sit on SSRN / bioRxiv / ChemRxiv without `from_arxiv` set** — a definition
  that requires `from_arxiv` misses all of them. It answers "is this an arXiv preprint",
  not "is this a preprint".
- Not one of the 137 arXiv rows has an empty venue, so a `NOT has_venue` clause never
  fires.

The repaired version is `venue_is_preprint` — regex the venue against the preprint servers,
no `from_arxiv` requirement (145 rows). The notebook measures **both**, and both sit at
chance (max distance 0.092 across six use cases).

**Recommendation:** drop. The narrow version is φ=0.83 with `from_arxiv`; the correct
version carries no signal.

### 2.5 `is_english` — no variance to work with

2,843 of 2,873 rows (**98.96%**) are `en`. The other 30 are spread across 12 languages, the
largest group being 9 Swedish rows. Full-population AUC is 0.491–0.503 across all six use
cases; Mann-Whitney reaches significance nowhere.

**Recommendation:** drop. If non-English papers ever matter, the honest handling is a
filter or a data-quality check at ingest, not a model feature.

### 2.6 `author_count` — real once, and for the wrong reason

Chance on five of six use cases (0.456–0.497). On `solar_leo` it scores 0.696 (p<0.001) —
genuinely interesting until you look at what else is true of that corpus:

- `solar_leo` is the **only corpus spanning 52 years** (1974–2026); the other five span 1–9.
- Its positives average 9.2 authors and year 2022.8; its negatives 4.5 authors and 2018.8.
- Plain `year` scores **0.766** there — better than `author_count`'s 0.696 — and the two
  correlate +0.22.

Author count is a weaker proxy for recency on the one corpus where recency separates
labels. Not leakage from the 30 seeded rows: excluding them, AUC holds at 0.693.

**Recommendation:** drop from the pooled build. Revisit `year` / `paper_age` (not author
count) at the per-use-case fine-tuning stage for corpora with a wide year span.

### 2.7 The venue cleanup — the task doesn't need to exist

"Clean out arXiv and long strings" assumes long venue strings are dirt. Of the 42 unique
venue strings over 100 characters, **40 name real conferences** ("proceedings of the 63rd
annual meeting of the association for computational linguistics…"). Exactly **2 are
genuinely dirty**.

Two bad rows out of 918 unique venues (0.1%) does not justify a filter that deletes 40
legitimate conference names — and the min-frequency threshold you need anyway for any
frequency or target encoding folds singletons into "other", which neutralises them for
free.

**Recommendation:** don't build it.

---

## 3. The features to build

### 3.1 Term overlap — the one that works

Already built and measured in `terms_overlap.ipynb`; nothing here supersedes it.

- `use_case_positive_terms` = `terms_must_include` + `terms_nice_to_have`,
  `use_case_negative_terms` = `terms_exclude`, both already broadcast onto every row.
- `term_overlap_positive` / `_negative` = whole-word regex match count against
  title + abstract.
- **Measured:** full-population ROC-AUC 0.801 (`cement_binders`), 0.709
  (`carbon_capture`), 0.698 (`ner`), all p<0.0001. Chance on the other three (0.46–0.56).
  On `carbon_capture` it **beats `relevance_score`** (0.709 vs 0.607).
- The hybrid surface-or-lemma spaCy variant (`terms_overlap_spacy.ipynb`) adds +0.010 on
  two use cases and rescues none of the dead three. Take it or leave it; plain regex is
  most of the value. **Do not use naive lemma matching** — it regressed `cement_binders`
  by −0.024 via a diagnosed spaCy POS-tagging quirk.
- `decision_must_have` / `decision_exclusions` add nothing (tested, not assumed).

**Known open issue:** `wf_pre_pipeline_checks.ipynb` already asked whether this feature is
really an abstract-length proxy. Carry that check forward before it goes in the pipeline.

### 3.2 Percentile ranks — pooled stage only

You're pooling for the ensemble build, then fine-tuning per use case. The percentile
features belong to exactly the first of those, and the notebook pins down why.

Per-use-case percentile ranking is a **monotonic transform within a use case**, and
ROC-AUC reads only ranks — so the per-use-case AUC is *identical to four decimal places*
for raw and percentile on all six use cases. The only number it changes in the entire
table is the pooled read (0.479 → 0.491).

That is not a strike against them; it defines what they are. The cross-use-case scale gap
is real and large — mean `citation_count` is 3.4 in `tech_forecasting` and 240.1 in
`solar_leo` — and a pooled model will otherwise learn "high citations ⇒ solar_leo ⇒
positive".

- **Pooled ensemble stage:** keep them. This is the stage they were designed for.
- **Per-use-case fine-tune:** drop them. They cannot change a single-use-case model's
  ranking, and carrying raw + percentile doubles the column count for nothing.

**Implementation requirement:** fit the percentile transform **inside each training fold**
and apply it to the held-out fold. A percentile computed over all rows peeks at the test
fold's distribution. The notebook's `fit_apply_percentiles()` is a working implementation
(train-fold ECDF via `np.searchsorted`).

**Caveat that outranks all of this:** normalising a feature that carries no signal
(0.479 → 0.491, both chance) gives a normalised feature that still carries no signal. These
are worth wiring up only for whichever numeric metadata survives §2 — on today's evidence,
possibly none of it.

---

## 4. Why `solar_leo` and `soil_microbiome` resist everything

Investigated in full in `notebooks/experiments/wf_hard_use_cases.ipynb`. Three
findings, and the third turned out to matter more than the question that prompted it.

### 4.1 The mechanism: the classes are interleaved in embedding space

For each labelled paper, take its 10 nearest neighbours by cosine similarity within its own
use case and measure how often they share its label — against the majority-class rate as
the do-nothing baseline:

| use case | k-NN agreement | majority baseline | lift |
|---|---|---|---|
| carbon_capture | 0.636 | 0.505 | **+0.131** |
| cement_binders | 0.760 | 0.649 | **+0.111** |
| tech_forecasting | 0.604 | 0.595 | +0.009 |
| ner | 0.699 | 0.715 | −0.015 |
| solar_leo | 0.703 | 0.767 | **−0.064** |
| soil_microbiome | 0.656 | 0.737 | **−0.080** |

That ordering is the ordering of every feature result in this repo. In the two hard corpora
a paper's nearest topical neighbours predict its label *worse than guessing the majority
class*. Embeddings, TF-IDF and term overlap all read topical similarity, so **no topical
feature can separate these labels.** The axis has to come from somewhere else.

### 4.2 `solar_leo`: the boundary is publication vintage, and the brief says so

Positive rate by year: 0.33 (pre-2018) → 0.63 (2018–20) → **0.98** (2020–22) → 0.86
(2024+). Plain `year` scores **0.766** against the embedding's 0.648 out-of-fold. And
topically identical papers sit on both sides of the boundary — *"Film Morphology Control
For High Efficiency Perovskite Solar Cells"* is negative (2015); *"Film Grain-Size Related
Long-Term Stability of Inverted Perovskite Solar Cells"* is positive (2016).

The use case brief explains it. The analyst's note says the corpus was *"seeded from the
reference lists of this field's review papers, so the starting pool is CANON — well-cited,
established work"*, and the `objective` asks for approaches that improve on incumbents
*"beyond what incumbent technologies currently deliver"*. Seed the pool with canon, then
reward going beyond canon, and old canonical papers are negative by construction.

**This is a labelling-design property, not a missing feature.** A model can reach 0.805
here on `year + author_count` alone, but it has learned this corpus's vintage distribution
and should not be expected to transfer. The corpus also carries outright retrieval noise
labelled negative (a 2011 spinal-cord neuroscience paper, a 1997 crystal-optics paper) —
a smaller, separate data-quality issue.

### 4.3 `soil_microbiome`: the axis is in `objective`, not in `terms.must_include`

Positives and negatives are near-indistinguishable by topic — both are soil-microbiome
crop papers. The split is **applied-intervention framing vs descriptive-ecology framing**:

| field | content |
|---|---|
| `terms.must_include` | soil microbial community, soil health, crop production, … *(pure topic — every paper has it, because the corpus was retrieved with it)* |
| `objective` | "research on microbial and fungal community **management strategies** that enhance…" |

A length-normalised two-column regex on that axis (intervention-word rate minus
descriptive-word rate) scores **0.718 full-population** (p = 3.6e-10, split-half stable at
0.722 / 0.719) and **0.727 out-of-fold — beating the 384-dim embedding's 0.712**, term
overlap's 0.504 and `relevance_score`'s 0.542.

**Caveat, stated plainly:** those regexes were written after reading ~16 labelled titles
from this corpus, so that number is optimistic. The transferable claim is not "use these
regexes" — it is that **`objective` and `problem_statement` carry the decision axis and no
feature in this repo currently reads them.**

### 4.4 The finding that outgrew the question: concatenation dilution

Both diagnosed features beat the embedding standing alone, yet adding them to it changes
almost nothing (`soil_microbiome` 0.712 → 0.713). That is not about these features — it is
what happens when one informative column is standardised alongside 384 embedding
dimensions on ~350 rows.

Compressing the embedding first fixes it, **on all six use cases**:

| use case | emb384 | emb384 + overlap | PCA32 | PCA32 + overlap | gain |
|---|---|---|---|---|---|
| carbon_capture | 0.732 | 0.738 | 0.796 | 0.813 | **+0.081** |
| soil_microbiome | 0.712 | 0.711 | 0.785 | 0.783 | **+0.071** |
| cement_binders | 0.799 | 0.798 | 0.864 | 0.865 | **+0.066** |
| tech_forecasting | 0.689 | 0.690 | 0.759 | 0.755 | **+0.066** |
| solar_leo | 0.648 | 0.653 | 0.701 | 0.705 | **+0.057** |
| ner | 0.700 | 0.705 | 0.723 | 0.731 | **+0.031** |

This reframes `reports/combined_features_notes.md`, which concluded that extra features
alongside a sentence embedding are "neutral-to-harmful". On this evidence that was a
finding about the *representation*, not the features — those experiments concatenated onto
the full-width embedding.

---

## 5. Actions for the ensemble

Ordered by measured impact on the number that matters — performance on a use case the
model has never seen. The pooled test below holds `tech_forecasting` out entirely and
scores it once:

| config | OOF dev | held-out `tech_forecasting` |
|---|---|---|
| emb384 (LR) | 0.772 | 0.520 |
| emb384 + overlap (LR) | 0.774 | 0.533 |
| PCA32 + overlap (LR) | 0.808 | 0.611 |
| PCA64 + overlap (LR) | 0.803 | 0.645 |
| emb384 + overlap (HGB) | **0.839** | 0.584 |
| **PCA32 + overlap (HGB)** | 0.820 | **0.714** |

### Action 1 — PCA-compress the embedding before anything is concatenated to it *(highest impact)*

The raw embedding transfers to an unseen use case at **0.520 — chance**. At 32 components
plus term overlap it reaches **0.714**. Fit the PCA **inside each training fold**; fitting
it on all rows leaks the test fold's covariance structure.

Tune `n_components` (32 and 64 both work; 64 was better under LR, 32 under HGB) as a real
hyperparameter, selected on held-out-use-case score.

### Action 2 — select on the held-out use case, not on dev OOF

Row 5 of that table is the trap: `emb384 + overlap` with gradient boosting has the **best
dev score in the table (0.839)** and transfers at 0.584. Choosing on dev OOF picks it and
loses 0.130 AUC on the unseen use case. Rotate the holdout across all six use cases rather
than trusting one, since `tech_forecasting` is the smallest corpus.

### Action 3 — build term overlap, and extend it to read `objective` / `problem_statement`

`term_overlap_positive` / `_negative` as specified in §3.1 — the only proposed feature with
a measured effect. Then extend it: §4.3 shows the discriminating vocabulary for
`soil_microbiome` lives in `objective`, which no current feature reads. Concretely, derive
a second term list from `objective` + `problem_statement` + `decision_must_have` (content
words, stopwords removed) and score overlap against it as a separate column.

This is the one genuinely new feature idea the investigation produced, and it is cheap.
Validate it on a corpus nobody has read before trusting the number.

### Action 4 — add `year` / `paper_age`, but as two raw columns and with eyes open

Not `citation_velocity` (§2.1). `year` carries real signal on the wide-span corpora
(`solar_leo` 0.766, `soil_microbiome` 0.577) and near-nothing elsewhere, and on `solar_leo`
it is partly the labelling artefact of §4.2. Include it, and check per-use-case feature
importance rather than assuming it means "recent papers are better".

### Action 5 — put provenance in as a negative control, never as a feature

Do not add `has_venue`, `citation_count_missing`, or `venue_is_arxiv_only`. Instead, track
whether held-out performance depends on the `from_*` columns: nullness is 100% determined
by which API answered (§2.2), so any model leaning on it has learned the retrieval
pipeline, not relevance. This is the most likely cause of a good CV score collapsing on a
new use case.

### Action 6 — treat `solar_leo` as a corpus problem, not a modelling one

Its labels encode "published recently enough to be beyond the incumbent". Options, in
increasing cost: exclude it from the pooled ensemble; keep it but never report it as
evidence the system finds relevant work; or re-label a sample against the decision criteria
without showing the labeller the publication year. Its own brief already warns the pool is
citation-biased canon — that warning should propagate into how its numbers are quoted.

### Not worth doing

Adding more metadata columns. Eight of them together are worth +0.002 (§1), and four
separate feature ideas have now been measured and rejected (TRL keywords, OpenAlex venue
quality, author ORCID, and this punch list). The headroom is in the representation
(Action 1) and in reading the parts of the use-case brief nobody reads yet (Action 3).

---

## 6. Where this evidence lives

| Claim | Source |
|---|---|
| Every number in §1–§3 | `notebooks/experiments/wf_feature_validation.ipynb` |
| Every number in §4–§5 | `notebooks/experiments/wf_hard_use_cases.ipynb` |
| Term-overlap AUCs | `notebooks/experiments/terms_overlap.ipynb` |
| spaCy variant, lemma regression | `notebooks/experiments/terms_overlap_spacy.ipynb` |
| Fold design reused throughout | `notebooks/experiments/wf_fold_pca_test.ipynb` |
| Original punch list this supersedes | `reports/wf_eda_fe_report.md` §4 |
| "Combined features don't help" — reframed by §4.4 | `reports/combined_features_notes.md` |
