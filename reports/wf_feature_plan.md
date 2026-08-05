# Feature engineering plan — the 11-feature punch list, after measurement

**Status:** decision document. Every number here is reproducible from
`notebooks/feature_experiments/wf_feature_validation.ipynb`, which builds all 11 proposed
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

## 4. What to do instead

The metadata track is close to exhausted. Four candidate features have now been tested and
rejected on measurement (TRL keyword estimate, venue quality via OpenAlex, author ORCID,
and now this list), which is a healthy trail — but the remaining headroom is elsewhere.

**The thread worth pulling:** three of six use cases resist *everything* tried so far.

| use case | term overlap | metadata | embedding (OOF) |
|---|---|---|---|
| cement_binders | 0.801 | 0.765 | 0.799 |
| carbon_capture | 0.709 | 0.593 | 0.775 |
| ner | 0.698 | 0.510 | 0.707 |
| soil_microbiome | 0.504 | 0.513 | 0.689 |
| solar_leo | 0.464 | 0.513 | **0.556** |
| tech_forecasting | 0.563 | — | held out |

`solar_leo` is the standout problem: term overlap is *below* chance and even the embedding
only reaches 0.556 out-of-fold. Its own analyst notes flag a citation-biased seed corpus,
and it is the one corpus where a metadata feature (recency) beats the embedding. That is a
labelling/corpus-construction question, not a feature-engineering one, and it is more
likely to move the project than a twelfth metadata column.

**Judgement call:** before adding features, work out why those three corpora behave
differently. `soil_microbiome` has the richest term list of any use case and still shows
nothing — that is a fact worth explaining.

---

## 5. Where this evidence lives

| Claim | Source |
|---|---|
| Every number in §1–§3 | `notebooks/feature_experiments/wf_feature_validation.ipynb` |
| Term-overlap AUCs | `notebooks/feature_experiments/terms_overlap.ipynb` |
| spaCy variant, lemma regression | `notebooks/feature_experiments/terms_overlap_spacy.ipynb` |
| Fold design reused here | `notebooks/modelling/wf_fold_pca_test.ipynb` |
| Original punch list this supersedes | `reports/wf_eda_fe_report.md` §4 |
