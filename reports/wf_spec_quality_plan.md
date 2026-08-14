# Spec quality — test plan for the four open angles

> **Status: PLANNED.** Pre-flight contract only, no probe has run. Every bar below is fixed
> **before** any measurement, per `CONTEXT.md` §5 and the pattern
> `reports/wf_llm_screening_plan.md` established.

**What this plans.** Four questions left open by `reports/wf_spec_quality_ablation.md` and
`reports/wf_label_budget_shape.md`, all of which feed the labelling UX brief
(`academic_agent: design/guided_funnel/DATA_BRIEF.md`).

| # | Probe | Cost | Blocks what |
|---|---|---|---|
| **P1** | The **reader** arm of the spec ablation | ~$5 | The spec linter needs two critics (D43) and only one is measured |
| **P2** | The **foreign-brief** brief-quality detector | **$0** | A free "your brief is broken" signal, computed and unused |
| **P3** | **Checkability** — is this use case automatable at all? | **$0** | A pre-scoring caption; S-UCQ Q5's TIRI-side half |
| **P4** | What **text the embedding sees** | Modal GPU, ~$3 | The last untested feature lever |

**Run order: P2, P3 (free, and they reframe the others) → P1 → P4 only if its gate passes.**

---

## 0. The discipline problem that shapes all four

🔴 **Set A is no longer a clean surface, and these probes must not pretend otherwise.**
`CONTEXT.md` §4 lists selection-on-holdout as an open risk: LOGO has been selected against many
times. Set A is now in the same position — the LLM screening run, the induced-brief ladder, the
ensemble swap, the label-budget grid and the spec ablation have all been measured on it, and the
last two were *mine*, this week.

So every probe below declares a surface up front:

| Surface | Status | Use for |
|---|---|---|
| `benchset_v1_large_set_a` | **burned** — 5+ selection passes | Exploration and effect-size estimation only |
| `benchset_v1_large_set_b` | clean | **Confirmatory runs.** Any bar below is read here |
| the 12 collections too small to split | untouched | P2 and P3, which need no train split |
| TIRI's own 6 use cases | burned for modelling, clean for *spec* questions | P3 — they are the only analyst-written specs available |

**Rule for all four probes: estimate on A, confirm on B, and report both.** A finding that
appears on A and not on B is a finding about A.

---

## P1 — The reader arm: do contradictions and fluff behave differently for an LLM?

### The question

`wf_spec_quality_ablation.md` measured 14 spec variants against a **matcher** (BM25 + term
overlap) and the cold-start prose feature. The reader was never measured, and the two are already
known to respond in opposite directions — term lists help matchers and cost readers recall
(`wf_llm_benchset_a_findings.md` §5). The one variant where this matters most is `conflicting`,
which cost the matcher **−0.014** while S-AL measured a written policy costing a *reader*
**0.828 → 0.772**.

> **H1.** For a reader, `conflicting` is the most damaging variant. For a matcher it was 10th of 13.
> **H2.** For a reader, `fluff_replace` and `vague_objective` are damaging at *every* label count,
> not only at cold start — because a reader consumes instructions, and there are none left.
> **H3.** `keyword_flood` is *less* damaging to a reader than to a matcher, or helps.

### Design

The existing harness, unchanged: `scripts/llm_pipeline_utils.py` `build_use_case_brief` is already
config-driven on `use_case_brief_cols`, and every response is disk-cached by (model, variant,
prompt version, params, prompt text). Reuse `run_spec_quality_ablation.variant()` to produce the
brief frames — **one function, both arms, so the two results are about the same 14 texts.**

- **Model:** `gemma-4-31b`, prompt **P2**. Not because it is best — it is the *weakest* of the four
  on set A — but because it is the model the whole B0/B1/B2 ladder ran on, so this joins that
  ladder rather than starting a new one. State that it is one family (caveat 3 of that report).
- **Variants:** all 14, zero-shot, so the brief is the only thing that varies.
- **Rows:** the same case-control sample and the same held-out rows as the matcher arm.
- **Metrics:** ROC-AUC **and** F2@own + fraction-read. The operating point is the thing a reader
  produces and a ranker cannot (`wf_llm_benchset_a_findings.md` §5), so a variant that holds AUC
  while collapsing recall is a failure this probe must be able to see.

### Bars, fixed now

| | Pass | Kill |
|---|---|---|
| **H1** | `conflicting` costs a reader **≥0.03 more AUC than it cost the matcher**, on ≥5 of 8 collections | `conflicting` is within 0.03 of the matcher's −0.014 → contradictions are cheap for everyone, and D43's three-critic linter collapses to two |
| **H2** | `fluff_replace` / `vague_objective` cost the reader **≥0.03** | inside the floor → prose damage is genuinely cold-start-only and the linter only needs to fire before the first labels |
| **H3** | `keyword_flood` costs the reader **less** than the matcher's −0.072, on ≥5 of 8 | it costs *more* → term-list damage is universal, which simplifies the guidance and is worth knowing |

### Cost

9,993 rows was $0.42 for `gpt-oss-20b` and ~$2 for gemma across a cell. **14 variants is too many
to buy at full width.** Stage it:

1. **Stage 1 (~$1.50):** the 4 decisive variants — `full`, `conflicting`, `fluff_replace`,
   `keyword_flood` — on a 2,000-row subsample of set A. Estimates only, no verdict.
2. **Stage 2 (~$3.50):** the same 4 on **set B**, full held-out rows. **The verdict is read here.**
3. Remaining 10 variants only if stage 2 shows the arms diverge at all.

**Ceiling $10.** Stop and report if stage 1 shows all four variants inside 0.03 of each other —
that is itself the answer (a reader is insensitive to spec form) and buying stage 2 adds nothing.

### Instrument checks

1. **Assert the brief text actually changed.** Print a hash of each variant's brief per collection
   and assert 14 distinct values. A config-driven column list fails silently if a column name is
   wrong — the variant would render identically to `full` and score identically, and it would look
   like a finding.
2. **Parse rate and tie fraction beside every number**, per `benchset_metrics`. Verbalised scores
   are lumpy (tie fraction 0.997–0.999) and WSS is soft on them.
3. **Reuse the shuffled-brief derangement as the floor.** It already returns AUC 0.498 / F2 0.000
   on this corpus, so it is a known-good calibration of "the reader is genuinely reading."

---

## P2 — The foreign-brief detector: a free "your brief is broken" signal

### The question

`notebooks/experiments/wf_usecase_diversity.ipynb` §6 built a full brief × corpus matrix — score
every use case's papers with every *other* use case's brief. It was built to test use-case overlap,
and it incidentally measured something nobody has used:

> `soil_microbiome` is screened **better by `solar_leo`'s brief (0.792) than by its own (0.603)**.
> `moran_2021` likewise.

**A brief that a stranger's brief outperforms on your own corpus is a broken brief.** That is a
computable, label-free-at-scoring-time diagnostic, it needs no LLM, and it is a `for` loop over
briefs already on disk.

> **H4.** `own_brief_auc − max(foreign_brief_auc)` identifies briefs that a labelled evaluation
> would independently call weak.

### Design

- Compute the full matrix on **all 34 use cases** (6 TIRI + 28 benchset), not the diversity
  notebook's subset. `usecase_diversity_utils` already has the loader.
- The statistic is the **margin**: `own − best_foreign`. Negative means broken.
- **Validation, and this is the part that makes it a probe rather than a plot:** the margin must
  predict something measured independently. Two candidates, both already on disk —
  (a) the collection's gain from labelling (`wf_label_budget_shape_grid.csv`), and
  (b) the collection's induced-brief improvement (`wf_llm_benchset_a_induced_features.md`), since a
  brief with room to improve is a brief that was weak.

### Bars, fixed now

| | Pass | Kill |
|---|---|---|
| **Specificity — the one that matters** | **≤2 of 34** use cases flag as broken. A detector that fires on healthy briefs trains the analyst to dismiss it — S-FR's own conclusion, and S-FF's generic-vocabulary failure is exactly how such a list fills up | ≥6 of 34 flag → it is measuring corpus difficulty, not brief quality. Report and stop |
| **Validity** | Spearman \|rho\| ≥ 0.5 against **at least one** of (a) or (b), and the sign is the predicted one | no relationship → the margin is a curiosity |
| **Confound control** | The margin must survive controlling for **prevalence**, which correlated with LOGO at rho −0.72 in the diversity notebook and is the obvious confound | it does not survive → report as confounded, do not ship |

### Cost

**$0.** All vectors and briefs are cached. ~10 minutes of compute.

### What it changes if it passes

`academic_agent` gains a **⑤ Rebuild** caption with an owner: *"three other use cases screen your
papers better than your own brief does — your objective may be describing your intent rather than
your literature."* That is `NUMBERS.md`'s rule satisfied — a number with a computation behind it.
It is also the only brief-quality signal that needs **no labels at all**, which makes it the only
one available at the moment the brief is written.

---

## P3 — Checkability: is this use case automatable, and can we say so before scoring?

### The question

This is **S-UCQ Q5**'s TIRI-side half (`academic_agent: spikes/use_case_quality/PLAN.md`), and it
is pre-registered there as **descriptive, not inferential** — with 4 use cases no correlation is
supportable. TIRI can supply the half S-UCQ cannot: the **evidence-availability** denominator,
across 34 use cases instead of 4.

The motivating observation is S-AL's: cement reaches AUC 1.000 under every model because its
criteria are checkable predicates (`>50% CO2 vs OPC`, `>=40 MPa @ 28d`); tech-intel has no
performance criteria at all and is the hardest use case in the repo.

But a checkable criterion is worthless if the abstracts do not state the evidence — and S-RS
measured **only ~10% of abstracts carry a number + unit.**

> **H5.** Two numbers, computable before any scoring, describe how automatable a use case is:
> **criterion checkability** (what fraction of the spec is an extractable predicate rather than a
> judgement invitation) and **evidence availability** (what fraction of the pool's abstracts state
> the evidence that predicate needs).

### Design

- **Deterministic, no LLM.** Regex for number+unit patterns, comparators, and the S-CE cue library.
- Report both figures per use case across all 34, with `n` beside each.
- Cross the two: a use case with high checkability and low evidence availability is the
  **diagnosable failure** — the spec is fine and the corpus cannot answer it.
- **Pre-registered as DESCRIPTIVE.** No correlation against AUC is claimed at n=34 with the
  heterogeneity `CONTEXT.md` §5 documents. The deliverable is the figure and the reasoning.

### Bar

| Pass | Kill |
|---|---|
| Both figures **vary materially across the 34** (IQR spanning ≥0.2) and the ordering is stable across two extraction rule sets | the figures do not discriminate → the metric does not ship, exactly as S-UCQ Q5 pre-registers |

### Cost

**$0**, no LLM, ~20 minutes.

### What it changes

A pre-scoring caption the funnel can show at ① or ⑤: *"your criteria are mostly judgement calls,
and 6% of these abstracts state a number — expect to label, not to automate."* It also gives
`NUMBERS.md` **N23** (criteria fields populated, currently 0 of 7 live use cases) a companion that
measures *quality* rather than presence.

---

## P4 — What text the embedding sees

### The question

`CONTEXT.md` §6's rejected list is a graveyard of **scalar** feature engineering: 11 metadata
features (+0.002), PCA, per-use-case centring, four third branches, the induced brief in the
ensemble (+0.001). All of them add columns *beside* the embedding. **None of them changed what text
goes into it** — and `embedding_utils.MODEL_CONFIGS` already carries `title_abstract_sep` as a
per-model quirk, so the seam exists and has never been treated as a variable.

> **H6.** The paper text fed to the embedder is an unexplored lever, and it is not the same
> question as adding features — it changes the representation rather than annotating it.

Variants: `title_only` · `title+abstract` (shipped) · `title+abstract+venue` · **`cue_sentences`**
(the ~3 solution-claim sentences S-CE's cue library extracts at 85.5–90.6% coverage — free,
deterministic, and orthogonal to embedding similarity).

### The gate — this probe is expensive and must be earned

**Do not run P4 until a $0 pre-check passes.** Re-embedding a corpus is Modal GPU time, and
`CONTEXT.md` §7 says route to Modal for genuinely heavy jobs, not on principle.

**Pre-check ($0):** the shipped export already carries **title-only** and **title+abstract**
material for the `cos_briefpre_*` / `cos_brief_*` column pairs on set A. If those two differ by
**<0.01 AUC**, the text-input axis is flat on the one comparison already paid for, and P4 is not
worth GPU time. **≥0.01 → proceed.**

### Bars, fixed now

| Pass | Kill |
|---|---|
| Any variant beats `title+abstract` by **≥0.03 within-silo AUC on ≥5 of 8 collections on set B** | all variants within 0.03 → the text input is not a lever, and the "features are already in the embedding" conclusion is complete rather than merely unrefuted |

### The trap to avoid, stated before running

**A label must record what text it was made against.** `wf_llm_benchset_a` caveat and
`ENSEMBLE_BRIEF` both name this: if the analyst labelled against an abstract and the model embeds
a cue-sentence extract, the training set silently mixes two conditions and an extraction error
becomes a laundered human label. `summaries.grounding` already exists in the sibling repo for this
reason. **Any `cue_sentences` arm that reaches a UI needs that stamp first.**

### Cost

Modal GPU, ~$3, 4 variants × 2 models on set B. Zero if the pre-check kills it.

---

## Sequencing, and what each result would change

```
P2 ($0) ─┬─► pass → ship a brief-quality caption with no labels required
         └─► fail → the only label-free brief signal is gone; P1 becomes the whole linter

P3 ($0) ─┬─► pass → a pre-scoring "expect to label, not automate" caption
         └─► fail → drop, exactly as S-UCQ Q5 pre-registers

P1 ($5) ─┬─► arms diverge → D43's two-critic linter is correct and buildable
         └─► arms agree  → one linter, one verdict, and the guidance simplifies

P4 (gated, ~$3) ─┬─► pass → the first new feature lever since the query-conditioned block
                 └─► fail → close the feature-engineering question for good
```

**The honest expectation, recorded so it can be wrong:** P2 passes on validity and **fails on
specificity** (detectors of this shape usually do); P3 passes descriptively; P1's H1 passes and H2
fails; P4's pre-check kills it. If that is how it lands, the useful output is a two-critic linter
plus a checkability caption, and the feature question closes — which is worth the ~$5 to know.

## What this plan does *not* cover

- **The label-noise ceiling** (24 unfilled adjudications in the sibling repo). It bounds every AUC
  in both repos including all four probes above, and it is human work, not a probe.
- **Prevalence-honest interface copy.** A design gap, not a measurement — no bar would be
  meaningful.

Both are detailed in the working notes rather than planned here, because neither is a probe.
