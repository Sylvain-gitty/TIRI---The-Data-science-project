# Feature engineering review — what we proposed, what we measured, what to build

**Audience:** anyone picking up TIRI's Week-2/3 work. Assumes no context beyond
`README.md`. Every number is reproducible from the two notebooks listed in §9.

**Scope:** covers a review session over the 11-feature metadata punch list from
`reports/wf_eda_fe_report.md` §4, plus a follow-up investigation into the two use cases
that fail every feature we try. Companion to `reports/wf_feature_plan.md` (the decision
doc); this one is the narrative.

---

## 1. ELI5 — what is this about?

> We have ~2,900 academic papers, spread across 6 research questions ("use cases"). A human
> analyst read each one and marked it **relevant** or **not relevant**. We want a model that
> can make that call automatically, so future research questions don't need a human to read
> hundreds of abstracts.
>
> To do that, each paper has to be turned into numbers a model can learn from. That's
> **feature engineering**. Someone had proposed a list of 11 numbers to compute — how many
> citations per year, how many authors, does it have a journal name, is it in English, and
> so on.
>
> **This review measured all 11 before building any of them.** Most turned out to be worth
> nothing, one was actively misleading, and the thing that helped most wasn't on the list at
> all — it was a change to how we handle the numbers we already had.

---

## 2. TL;DR

Three findings, in order of how much they matter.

**① Almost none of the proposed metadata features work.** All of them together are worth
**+0.002 AUC**. Worse, they *improve* the score on the data we develop against while
*lowering* it on an unseen use case — the signature of learning our own search pipeline
instead of learning relevance.

**② Two use cases (`solar_leo`, `soil_microbiome`) resist every feature for a diagnosable
reason.** Their relevant and irrelevant papers are about the *same topics*, so anything
that measures topic is blind to them. In one case the real dividing line is publication
year; in the other it's a distinction written in a part of the use-case brief that no
feature currently reads.

**③ The biggest win had nothing to do with new features.** Compressing the paper embedding
from 384 numbers down to 32 before adding anything to it takes performance on an unseen use
case from **0.520 (a coin flip) to 0.714**. This also explains an earlier negative result in
this repo, which had been attributed to the features rather than the setup.

> **ELI5 on "AUC":** a score from 0 to 1 for how well a number separates two groups. **0.5
> means useless** — you'd do as well flipping a coin. 1.0 is perfect. 0.7 is genuinely
> useful. Below 0.5 means it's *backwards* (the thing you thought predicted "relevant"
> actually predicts "not relevant").

---

## 3. How we worked, and why it matters

The punch list in `wf_eda_fe_report.md` §4 came from exploratory analysis — looking at
averages, missing-value rates and correlations. That's a good way to generate ideas. It is
not evidence that an idea works.

So the first step was not to build the features. It was to **build all 11 and measure
each one against the labels we already have.** That took one notebook and turned a
build-list into a mostly-don't-build list.

> **ELI5 on why the order matters:** it's the difference between "papers we marked relevant
> have more citations on average, so citations should help" and "when we actually rank
> papers by citations, does the ranking match the labels?" The first can be true while the
> second is false — one paper with 26,000 citations drags an average around, but it's still
> just one paper. That is exactly what happened here.

**Two rules we held to** (both from `CLAUDE.md`):

- **NULL is not 0.** A missing citation count means "we never looked it up", not "it has
  zero citations". These are different facts and we never filled one in as the other.
- **Cross-validated, never train-then-score.** Every model number below is on data the
  model never saw during fitting. Where a number is a plain measurement with no model
  involved, it says so.

---

## 4. Part 1 — the 11 proposed features, measured

### 4.1 The verdict

| Feature | What it was meant to capture | Verdict |
|---|---|---|
| `term_overlap_positive` / `_negative` | Does the paper use the words the analyst said mattered? | ✅ **Build** — the only one with a real effect |
| `per_use_case_percentile_*` | "High for this field" vs "high in absolute terms" | ✅ **Build, pooled stage only** |
| `citation_velocity` | Citations per year, to undo the age advantage of old papers | ❌ Doesn't fix what it was for |
| `author_count` | Big collaborations signal serious work | ❌ Coin flip on 5 of 6 use cases |
| `has_venue` | Published in a real journal vs not | ❌ Measures our search tool, not the paper |
| `venue_is_arxiv_only` | Preprint that never got published | ❌ Duplicates a column we already have |
| `is_english` | Language filter | ❌ 98.96% of rows are one value |
| venue "clean out long strings" | Strip junk from the journal field | ❌ Would delete 40 real venues to remove 2 bad ones |

### 4.2 The three failures worth understanding

**`citation_velocity` — the fix didn't fix it.** The reasoning was sound: old papers have
had longer to accumulate citations, so divide by age. But after dividing, **five of six use
cases still score below 0.5** — more citations still means *less* likely to be relevant.

> **ELI5:** a famous, heavily-cited paper can still be completely off-topic for a narrow
> question. Fame and fit are different things, and in this data they point in opposite
> directions.

The original evidence for this feature was a difference in averages. When we ranked papers
instead of averaging them, the effect disappeared — it had been driven by outliers (one
paper in this corpus has 26,506 citations).

**`has_venue` — it measures our own plumbing.** This is the most important cautionary
finding in the review. Whether a field is empty turns out to be **entirely determined by
which search API found the paper**:

| Source | Rows | `citation_count` missing | `venue` missing |
|---|---|---|---|
| arxiv | 135 | **100%** | 0% |
| core | 66 | **100%** | **100%** |
| crossref | 977 | 0% | 22% |
| europe_pmc | 404 | 0% | **100%** |
| openalex | 895 | 0% | 9% |
| pubmed | 180 | **100%** | 0% |

> **ELI5:** Europe PMC's API simply doesn't return a journal name. PubMed doesn't return
> citation counts. So "this paper has no journal name" doesn't tell you anything about the
> paper — it tells you which website we found it on. Different research questions send
> queries to different websites, so this quietly becomes "which use case is this?" in
> disguise.

And it behaves exactly like a confound should: it points **one way on one use case and the
opposite way on another** (0.570 on `carbon_capture`, 0.412 on `solar_leo`, both
statistically significant). A feature whose direction depends on which corpus you're
looking at isn't a signal.

**The trap this opens.** A feature nobody proposed — "is the citation count missing?" —
scores **0.238** on `cement_binders`, making it the strongest metadata reading in the whole
table. It's real, and it's entirely an artefact of which API answered. §6.3 shows this is
not hypothetical: including metadata measurably damages performance on an unseen use case.

**The percentile features — inert, but not useless.** Ranking a paper's citations against
others in its own use case is a *monotonic* transform, and our scoring metric only reads
rank order. So per-use-case AUC is **identical to four decimal places** with and without it.

> **ELI5:** re-labelling the runners in a race as 1st, 2nd, 3rd doesn't change who won. It
> only matters when you're comparing runners from *different races* — which is exactly what
> pooling all 6 use cases does.

Since the plan is to pool first and fine-tune per use case afterwards, these earn their
place at the pooled stage and should be dropped at the fine-tuning stage. The scale gap
they exist to fix is real: mean citation count is 3.4 in `tech_forecasting` and 240.1 in
`solar_leo`.

---

## 5. Part 2 — why two use cases resist everything

Across three separate feature efforts, the same use cases kept failing.

### 5.1 The mechanism

For each paper we took its 10 most similar papers (within its own use case) and asked how
often they carry the same label. The benchmark is "always guess the most common label":

| Use case | Agreement with neighbours | Benchmark | Difference |
|---|---|---|---|
| carbon_capture | 0.636 | 0.505 | **+0.131** |
| cement_binders | 0.760 | 0.649 | **+0.111** |
| tech_forecasting | 0.604 | 0.595 | +0.009 |
| ner | 0.699 | 0.715 | −0.015 |
| solar_leo | 0.703 | 0.767 | **−0.064** |
| soil_microbiome | 0.656 | 0.737 | **−0.080** |

> **ELI5:** normally, papers about the same thing get the same verdict — so if you don't
> know a paper's label, looking at its closest neighbours is a good guess. For these two use
> cases that's *worse than useless*: a paper's nearest neighbours actively point the wrong
> way. Relevant and irrelevant papers are mixed together on the topic axis.

Every feature we'd tried — embeddings, TF-IDF, term overlap — measures topic. If topic
doesn't separate the labels, none of them can. **This isn't a feature problem; it's a
"we're measuring the wrong dimension" problem.**

### 5.2 `solar_leo` — the dividing line is publication year

The share of papers marked relevant, by year: 0.33 before 2018 → 0.63 in 2018–20 → **0.98**
in 2020–22. Plain publication year scores **0.766**, better than the 384-dimension
embedding's 0.648.

Nearly identical papers sit on opposite sides:

- *not relevant*, 2015: "Film Morphology Control For High Efficiency Perovskite Solar Cells"
- *relevant*, 2016: "Film Grain-Size Related Long-Term Stability of Inverted Perovskite Solar Cells"

The use-case brief explains it. The analyst's own note says the corpus was *"seeded from the
reference lists of this field's review papers, so the starting pool is CANON — well-cited,
established work"*, and the stated objective is to find approaches that improve *"beyond
what incumbent technologies currently deliver."*

> **ELI5:** they filled the pool with the field's greatest hits, then asked "what's better
> than the greatest hits?" Older classic papers are the thing being improved on, so they get
> marked not-relevant almost by definition. The model isn't learning what makes a paper
> relevant — it's learning what year it came out.

**This is a corpus-design issue, not a missing feature.** A model can score 0.805 here using
year and author count, but that rule is about this corpus's history, not about relevance,
and it won't transfer. The corpus also contains obvious retrieval noise marked not-relevant
(a 2011 spinal-cord neuroscience paper, a 1997 crystal-optics paper) — a smaller, separate
data-quality problem.

### 5.3 `soil_microbiome` — the dividing line is in a field nobody reads

Here the papers really are hard to tell apart:

- *relevant*: "The Role of Arbuscular Mycorrhizal Fungi in Enhancing Crop Resilience and Soil Health: A Review"
- *not relevant*: "Microbial Products and Their Role in Soil Health and Sustainable Agriculture"

The split is between papers about **doing something** (inoculants, amendments, engineering,
management) and papers about **describing something** (community assembly, diversity,
mechanisms, biogeochemistry). Now compare two parts of the use-case brief:

| Field | Content |
|---|---|
| `terms.must_include` | soil microbial community, soil health, crop production, … — *pure topic words, which every paper in the corpus has, because those words are what retrieved them* |
| `objective` | "research on microbial and fungal community **management strategies** that enhance…" |

> **ELI5:** the search terms describe the *subject* — and everything we found is on that
> subject, so they can't tell the good from the bad. The thing that actually decides the
> verdict ("management strategies", i.e. does this paper *do* something) is written in a
> different box on the form, and none of our features read that box.

A two-column word-counting feature built on that distinction scores **0.727** — beating the
384-dimension embedding's 0.712, term overlap's 0.504, and `relevance_score`'s 0.542.

**Honest caveat:** those word lists were written *after* reading about 16 labelled titles
from this corpus, so that number is optimistic. The transferable claim is not "use these
word lists" — it's that **`objective` and `problem_statement` contain the deciding
information and no feature currently reads them.**

---

## 6. Part 3 — the finding that outgrew the question

### 6.1 The puzzle

Both diagnosed features beat the embedding **on their own**. Yet adding them *to* the
embedding changed almost nothing — `soil_microbiome` went 0.712 → 0.713.

> **ELI5:** a paper's embedding is 384 numbers. Adding one more useful number gives you 385,
> and the model spreads its attention across all of them — so the good one gets about
> 1/385th of the say. It's not that the feature is bad. It's that it got drowned out.

### 6.2 The fix, and it applies everywhere

Compress the 384 numbers down to 32 first, *then* add the feature. This helps on **all six**
use cases:

| Use case | Raw embedding | Raw + overlap | Compressed + overlap | Gain |
|---|---|---|---|---|
| carbon_capture | 0.732 | 0.738 | 0.813 | **+0.081** |
| soil_microbiome | 0.712 | 0.711 | 0.783 | **+0.071** |
| cement_binders | 0.799 | 0.798 | 0.865 | **+0.066** |
| tech_forecasting | 0.689 | 0.690 | 0.755 | **+0.066** |
| solar_leo | 0.648 | 0.653 | 0.705 | **+0.057** |
| ner | 0.700 | 0.705 | 0.731 | **+0.031** |

**This reframes an earlier conclusion.** `reports/combined_features_notes.md` found that
extra features alongside a sentence embedding are "neutral-to-harmful". On this evidence
that was a finding about the *setup*, not the features — those experiments concatenated onto
the full-width embedding, where nothing can help.

### 6.3 The number that actually matters

The point of TIRI is triaging a research question the model has **never seen**. So the real
test holds one use case out completely and scores it once at the end. (`relevance_score` is
excluded throughout — see `wf_ensemble_report.md` §0, which disqualified it as a feature.)

| Config | Score on dev data | **Score on unseen use case** |
|---|---|---|
| raw embedding | 0.772 | 0.520 |
| raw embedding + metadata | 0.773 | 0.504 |
| raw embedding + term overlap | 0.774 | 0.533 |
| compressed + term overlap (linear) | 0.808 | 0.611 |
| raw embedding + term overlap (boosted) | **0.839** | 0.584 |
| **compressed + term overlap (boosted)** | 0.820 | **0.714** |
| compressed + term overlap + metadata | 0.828 | 0.666 |

**Three things to read off this table:**

1. **The raw embedding transfers at 0.520 — a coin flip.** Compressed, with term overlap, it
   reaches 0.714.
2. **Row 5 is a trap.** It has the *best* dev score in the table and transfers 0.130 worse
   than the recommended config. **Picking a model on the development score selects the wrong
   model here.**
3. **Metadata makes the dev score go up and the real score go down** (rows 6 vs 7: dev +0.008,
   unseen −0.048; and rows 1 vs 2). *Judgement call — this is one held-out use case of 264
   rows, so read it as directional.* But it is exactly what §4.2's provenance confound
   predicts: the model is learning which search API answered, which doesn't carry over.

---

## 7. Recommended actions

1. **Compress the embedding (PCA) before concatenating anything to it.** Biggest single win
   available: 0.520 → 0.714 on an unseen use case. Fit the compression *inside each training
   fold* — fitting it on all rows leaks information from the test fold. Treat the number of
   components as a real hyperparameter (32 and 64 both worked).
2. **Select models on held-out-use-case score, not development score.** See trap above.
   Rotate the held-out use case across all six rather than trusting one.
3. **Build term overlap, then extend it to read `objective` and `problem_statement`.** The
   only proposed feature with a measured effect, and §5.3 shows the deciding vocabulary
   often sits in fields we don't currently read. This also directly improves the
   **zero-label cold start**, which per `wf_ensemble_report.md` §4 can only use unsupervised
   ranking — embeddings plus term overlap.
4. **Add `year` / `paper_age` as two raw columns** — not `citation_velocity`. Check
   per-use-case importance rather than assuming "newer is better"; on `solar_leo` it's
   partly the labelling artefact from §5.2.
5. **Use provenance (`from_*`) as a negative control, never a feature.** If held-out
   performance depends on it, the model has learned the search pipeline.
6. **Treat `solar_leo` as a corpus problem.** Either exclude it from the pooled build, or
   keep it but never cite it as evidence the system finds relevant work. Its own brief
   already warns the pool is citation-biased canon.

**Not worth doing:** more metadata columns. Four separate ideas have now been measured and
rejected (TRL keyword estimate, OpenAlex venue quality, author ORCID, and this punch list).
The headroom is in the representation (action 1) and in reading the parts of the use-case
brief nobody reads yet (action 3).

---

## 8. Limits of this review — what we did *not* establish

- **The `soil_microbiome` word lists are biased.** Written after reading part of that
  corpus. The *method* (derive the axis from `objective`) is untested on a fresh corpus.
- **One held-out use case.** `tech_forecasting` is the smallest corpus (264 labelled rows).
  Every "unseen use case" number rests on it until the holdout is rotated.
- **Feature interactions weren't searched.** Each feature was measured alone, then all
  together. A metadata feature that only works in combination with another would have been
  missed.
- **We didn't retest the term-overlap feature itself.** Its numbers are carried over from
  `terms_overlap.ipynb`. A known open question there — whether it's partly an
  abstract-length proxy — is still open.
- **No causal claim about the labels.** We can show `solar_leo`'s labels track publication
  year; we can't show *why* the analyst labelled that way beyond what the brief states.

---

## 9. Where the evidence lives

| What | Where |
|---|---|
| The 11 features built and measured | `notebooks/feature_experiments/wf_feature_validation.ipynb` |
| Why the two use cases resist; the compression finding | `notebooks/feature_experiments/wf_hard_use_cases.ipynb` |
| Decisions in checklist form | `reports/wf_feature_plan.md` |
| Term-overlap evidence (carried over, not retested) | `notebooks/feature_experiments/terms_overlap.ipynb`, `terms_overlap_spacy.ipynb` |
| Fold design reused throughout | `notebooks/modelling/wf_fold_pca_test.ipynb` |
| The punch list this review supersedes | `reports/wf_eda_fe_report.md` §4 |
| `relevance_score` disqualification; cold-start ladder | `reports/wf_ensemble_report.md` §0, §4 |
| "Combined features don't help" — reframed by §6.2 | `reports/combined_features_notes.md` |

Both notebooks are read-only, write nothing to `data/processed/`, and run from their own
folder against `data/processed/papers_combined.parquet`.
