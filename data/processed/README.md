# Dataset overview: `papers_combined.parquet`

This document describes a dataset of academic papers, each evaluated for relevance
to one of six specific research questions. It's written to stand on its own — no
other files, code, or context should be needed to understand what's in it.

## Summary

Each of six research teams defined a research question they wanted the academic
literature searched for (e.g. "find alternative cement binder chemistries that cut
embodied carbon by more than half"). For each question, a search was run against
public academic metadata sources (Crossref, OpenAlex, arXiv, PubMed, Europe PMC,
CORE, Semantic Scholar), candidate papers were scored for relevance, and a human
analyst reviewed a portion of them, judging each as relevant, not relevant, or
borderline.

This dataset combines the results of all six research questions into a single
table: one row per paper, carrying that paper's bibliographic details, its
relevance judgement (where one exists), a precomputed vector embedding of its
text, and the original research question's full brief (objective, required/
excluded search terms, technology-maturity constraints, decision rules) repeated
onto every row it applies to.

It's a **snapshot**, not a live feed — each research question's data was collected
at a different point in time (see `exported_at` below), and the underlying
searches could be re-run and would likely return different/updated results.

## Shape

| | |
|---|---|
| Rows | 2,873 (one per paper) |
| Columns | 50 |
| File size | ~8.85 MB (Parquet, a compressed columnar format) |
| Research questions covered | 6 |

Row count per research question:

| Code | Research question | Papers |
|---|---|---:|
| `cement_binders` | Low-carbon cement binders | 690 |
| `soil_microbiome` | High quality microbial and fungal community in Soil | 602 |
| `ner` | Named Entity Recognition | 541 |
| `solar_leo` | Solar Cells for Low Earth Orbit Satellites | 427 |
| `carbon_capture` | Carbon Capture | 323 |
| `tech_forecasting` | Predicting Technology Developments | 290 |

## Structure: one row = one paper

Every row is a single paper evaluated against a single research question (a paper
considered for two different research questions would appear as two separate
rows). Columns fall into four groups, described below.

### 1. Paper identity

| Column | Type | Description |
|---|---|---|
| `paper_id` | string | The unique key for this row across the whole dataset. Format: `<research_question_code>__<source_id>`, e.g. `carbon_capture__3355`. |
| `use_case_key` | string | Short code for the research question — one of the six codes in the Shape table above. |
| `use_case_name` | string | Human-readable name of the research question, e.g. "Carbon Capture". |
| `source_id` | integer | The paper's identifier within its own research question's original data. **Not unique across the whole dataset** — two different research questions can reuse the same `source_id` for two different papers. Always use `paper_id` to refer to a specific row. |

### 2. Bibliographic metadata

| Column | Type | Description |
|---|---|---|
| `title` | string | Paper title. Never missing. |
| `authors` | string | Comma-separated author list. Missing for 26 of 2,873 rows. |
| `year` | nullable integer | Publication year. Range 1974–2027 (a few 2027 values are legitimate in-press/forthcoming papers, not errors). Missing for 15 rows. |
| `venue` | string | Journal/conference/publisher name, lowercased for consistency. Missing for 49 rows. |
| `doi` | string | Digital Object Identifier. Missing for 176 rows — this means no DOI-indexing source had a record for that paper, not that a duplicate was found and removed. |
| `url` | string | Link to the paper. Missing for 14 rows. |
| `sources` | string | Which metadata provider(s) matched this paper, comma-joined, e.g. `"crossref, openalex"`. Never missing. |
| `from_arxiv`, `from_core`, `from_crossref`, `from_europe_pmc`, `from_openalex`, `from_pubmed`, `from_seed`, `from_semantic_scholar` | boolean | One flag per metadata provider (derived from `sources`), so "did this come from arXiv" is a direct filter rather than a string search. |
| `language` | string | Language code of the title/abstract. 2,843 of 2,873 rows are `en`; the remaining 30 span 12 other languages. |
| `citation_count` | nullable integer | Citation count at the time of collection. **A missing value here means the count was never found — it is not the same as a citation count of zero**, which means a count was found and it was zero. Missing for 410 rows; where known, ranges from 0 to 26,506 (a small number of very highly cited papers pull this range up — citation counts are always heavily skewed like this). |

### 3. Relevance judgement & score

| Column | Type | Description |
|---|---|---|
| `triage_label` | categorical: `positive` / `negative` / `pass` | A human analyst's judgement on whether the paper is relevant to its research question. **A missing value means the paper was never reviewed by a human at all** — a different fact from `pass`, which means an analyst looked at it and judged it borderline. 478 of 2,873 rows have never been reviewed. |
| `review_label` | categorical: `soft_pos` / `soft_neg` / `hard_pos` / `hard_neg` | A second, more detailed review-stage judgement, applied after `triage_label`. Populated for only 1 row in the entire dataset — this second review pass has barely started across any of the six research questions, so don't build anything that assumes this column is usable yet. |
| `relevance_score` | decimal number | An automated relevance score comparing the paper to its research question. Roughly, but not strictly, bounded between 0 and 1 (the observed range in this dataset is -0.10 to 0.85). This is a machine-computed ranking signal, separate from `triage_label`, which is the human's judgement. |

### 4. Paper text & embedding

| Column | Type | Description |
|---|---|---|
| `abstract` | string | The paper's abstract. Empty (zero-length) for a small number of title-only records — see `has_abstract`. |
| `has_abstract` | boolean | `True` if `abstract` contains at least one word. Use this to filter out title-only rows without checking string length yourself. |
| `embedding` | array of 384 numbers | A precomputed sentence-embedding vector for the paper's title + abstract — a numeric representation of the text's meaning, ready to use as input to a machine-learning model. |
| `embed_model` | string | Which model produced `embedding`. Identical for every row in this dataset: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (with a version suffix). |
| `embed_dim` | integer | Length of the `embedding` vector. `384` for every row. |
| `exported_at` | timestamp | When this paper's research question was collected/exported. Varies by research question — this is not a single dataset-wide collection date. |
| `app_version` | string | Version of the tool that produced the export. |

### 5. Research-question context

These columns hold the analyst's original research brief for the paper's research
question — identical for every paper within the same research question, repeated
onto every row so a paper's context travels with it without needing a separate
lookup table.

| Column | Type | Description |
|---|---|---|
| `problem_statement` | string | The real-world problem motivating this research question, in the analyst's own words. |
| `objective` | string | What a good answer to this research question would look like. |
| `usecase_schema_version` | string | Version of the research-brief format used (currently `"1.0"` for all rows). |
| `domain_industry`, `domain_application` | string | Which industry/application area this research question belongs to. Sometimes blank — not every research question specifies both. |
| `domain_technology_focus` | list of strings | Named technologies or subfields this research question is about. |
| `terms_must_include`, `terms_nice_to_have`, `terms_exclude` | list of strings | Search terms the analyst specified as required, desirable, or disqualifying. |
| `performance_criteria` | list of {metric, target} pairs | Quantified success criteria, where specified, e.g. `{"metric": "embodied CO2 reduction", "target": ">50% vs OPC"}`. Empty for 3 of the 6 research questions. |
| `trl_min`, `trl_max` | nullable integers | Technology Readiness Level window of interest (1 = basic research, 9 = proven in its final, real-world form). Both are missing together for 3 of the 6 research questions that didn't specify a TRL window (1,215 of 2,873 rows). |
| `constraints_scale`, `constraints_cost` | string | Free-text scale/cost constraints, where specified. `constraints_scale` is blank for every single row in this dataset — no research question currently uses this field. |
| `decision_must_have`, `decision_nice_to_have`, `decision_exclusions`, `decision_rules` | list of strings | The analyst's explicit inclusion/exclusion rules used when triaging papers for this research question. |
| `notes` | string | Free-text notes from the analyst. Often blank. |

## A worked example

One row, trimmed for length (the real `abstract` is several sentences, and
`embedding` has 384 numbers, not 3):

```
paper_id:             carbon_capture__3355
use_case_key:          carbon_capture
use_case_name:         Carbon Capture
title:                 Optimization of amine-based carbon capture: Simulation and
                       energy efficiency analysis of absorption section
year:                  2024
venue:                 results in engineering
doi:                   10.1016/j.rineng.2024.103574
citation_count:        21
triage_label:          positive
relevance_score:       0.7995
abstract:              "Achieved over 98% CO2 removal efficiency with a minimum
                       energy consumption of 6035 KW using amine-based
                       absorbents in post-combustion carbon capture. ..."
embedding:             [-0.2197, 0.3606, 0.0849, ... ]  (384 numbers total)
embed_model:           sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
objective:             Find amine-based carbon capture approaches that operate at
                       lower temperatures without high chemical input or clean-up
                       costs.
terms_must_include:    [amine, carbon capture, low temperature, solvent regeneration]
trl_min / trl_max:     (not specified for this research question)
decision_exclusions:   [Solid-supported or solid sorbent amine systems (no liquid
                       solvent phase), Process optimization or simulation studies
                       that do not report a novel solvent, material, or
                       regeneration method, ...]
```

## Things to check before relying on this data

- **`review_label` is essentially unpopulated** (1 row out of 2,873) — treat any
  analysis of this column as not yet meaningful.
- **Label balance varies a lot by research question.** For one research question,
  `positive` is the *minority* label among reviewed papers; for others it's the
  large majority. Don't assume a uniform base rate across research questions.
- **A missing value is not the same as zero or "not applicable."** Throughout this
  dataset, a missing `citation_count`, `triage_label`, `trl_min`, etc. specifically
  means "this was never determined" — collapsing it into zero or false would
  silently change what the column means.
- **This is a point-in-time snapshot.** Different research questions were
  collected on different dates (see `exported_at`), and a small number of papers
  carry a `year` after the current year (legitimate in-press records).
