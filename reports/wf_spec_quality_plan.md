# Spec quality — test plan for the four open angles

> **Status: RUN AND CLOSED, 2026-08-14.** All four probes executed; see
> [Outcome](#outcome--2026-08-14-all-four-run) at the foot of this file for the scoreboard. The
> pre-flight contract and its Amendment are preserved below unchanged. Every bar was fixed
> **before** any measurement, per `CONTEXT.md` §5 and the pattern
> `reports/wf_llm_screening_plan.md` established.
>
> ⚠️ **Read [the Amendment](#amendment--2026-08-14-before-any-probe-ran) before any bar below.**
> Three of the four probes were pre-registered against premises that do not hold, and the
> corrected bars live there. Superseded text is kept struck through rather than deleted.

**What this plans.** Four questions left open by `reports/wf_spec_quality_ablation.md` and
`reports/wf_label_budget_shape.md`, all of which feed the labelling UX brief
(`academic_agent: design/guided_funnel/DATA_BRIEF.md`).

| # | Probe | Cost | Blocks what |
|---|---|---|---|
| **`P-R`** (was P1) | The **reader** arm of the spec ablation | ≈$4.10 measured (~~~$5~~) | The spec linter needs two critics (D43) and only one is measured |
| **`P-FB`** (was P2) | The **foreign-brief** brief-quality detector | **$0** | A free "your brief is broken" signal, computed and unused |
| **`P-CK`** (was P3) | **Checkability** — is this use case automatable at all? | **$0** | A pre-scoring caption; S-UCQ Q5's TIRI-side half |
| **`P-TX`** (was P4) | What **text the embedding sees** | **$0** to gate, Modal GPU ~$3 only if it passes | The last untested feature lever |

**Run order: `P-FB`, `P-CK` (free, and they reframe the others) → `P-R` → `P-TX` only if its gate passes.**

> **Why the probes were renamed.** `P1`–`P4` collided with the prompt variants `P1`/`P2`/`P4` *and*
> with `CONTEXT.md` §6's rejected "**P4**, the per-criterion checklist prompt". This document already
> tripped over it below — "Model: `gemma-4-31b`, prompt **P2**" sits inside probe **P1**. Three
> namespaces, one letter. `KILL` is likewise replaced by **`FAIL`**, which is house style
> (`wf_llm_benchset_a_findings.md` §1); an `inside floor` column is kept, because "inside the floor"
> is a distinct outcome from "worse" and three of these four probes have it as their fail condition.

---

## Amendment — 2026-08-14, before any probe ran

**Why this exists.** Reading the code the four probes were going to reuse found that **three of the
four were pre-registered against premises that are false**, and that one piece of infrastructure all
four depend on does not exist. Amending a pre-registration *before* any measurement is legitimate and
is what §5's discipline requires; amending it *after* would not be. Nothing has run, so this window is
open now and closes the moment the first probe executes. Every correction below is a correction to a
**premise**, not to a threshold that turned out inconvenient.

| # | The plan assumed | What is actually true |
|---|---|---|
| 1 | "confirm on `set_b`" is a step | **No set-B loader exists.** `grep set_b scripts/*.py` → zero hits, and nothing in the repo has ever opened `benchset_v1_large_set_b.parquet`. It is also **7 collections, not 8**, so every "≥5 of 8" bar is unreadable there |
| 2 | `P-FB` covers "all 34, not the diversity notebook's subset" | `wf_usecase_diversity.ipynb` §6 **already ran all 34**. What `P-FB` adds is validation, not coverage — and **7 of 34 margins are already printed in that notebook**, so its specificity bar cannot be read blind |
| 3 | `P-CK` measures "what fraction of the spec is an extractable predicate" | **No criteria field is populated across 34.** `performance_criteria` 3/34, `decision_criteria` 5/34, `constraints.trl` 3/34. There is no denominator, so an IQR over 34 is not computable |
| 4 | `P-TX`'s $0 gate compares title-only against title+abstract | `cos_briefpre_*` is a **brief-field** ablation (`embed_benchsets.BRIEF_VARIANTS["pre_screening"]`), not a paper-text one — the paper vectors are byte-identical between the pair. That comparison **already ran** at a paired median of 0.003–0.007 (`03_eda_full_benchset_v1.ipynb` §8.4), below the 0.01 gate. Running it as written would have **failed `P-TX` on a number about the wrong axis** |
| 5 | `build_use_case_brief` is "already config-driven on `use_case_brief_cols`" | True, but that is the **layer-1 notebook** function; `use_case_brief_cols` appears nowhere in `scripts/`. The screening harness uses `render_brief`, hardcoded to `BRIEF_FIELDS` and pinned by `BRIEF_RENDER_VERSION`. The config-driven-ness this plan relies on comes from `variant()` rewriting the **columns**, which does work — but for a different reason than stated |

### What changes, probe by probe

**`P-R`.** Bars move from "≥5 of 8" to **≥5 of 7** (set B's real width). Cost is corrected upward: the
measured gemma unit cost is $0.000132–0.000151/row, and set B's case-control sample is 11,011 rows, so
4 variants at full width is ≈$6.2 and all 14 is ≈$21.6 — over the ceiling. **Stage 2 therefore scores
held-out rows only** (`split != "train"`, ≈4,400), which is also the population the matcher arm scored
and so the comparable one, not merely the cheap one. Total ≈**$4.10**. The remaining 10 variants are
**explicitly out of budget** unless the ceiling is raised.

Three additions this plan needed and did not have:

1. 🔴 **The matcher arm must be re-run on set B, and it is free.** H1 and H3 compare a reader delta to
   a matcher delta, but the matcher's −0.014 / −0.072 were measured on set **A**'s 8 collections.
   Comparing a reader number on B to a matcher number on A is the selection-on-holdout error in
   miniature. `run_spec_quality_ablation.py` gains a `--set` flag; the re-run costs $0 and minutes.
2. 🔴 **`full` must be scored with `brief_map=None`.** `score_frame` appends the brief tag *only* when
   `brief_map is not None`, and gemma's set-A `P2` own-brief cell is already paid for under tag
   `P2|P2-v1|brief-v1`. Routing `full` through `brief_map=` with any tag changes the cache key and
   **re-buys ≈$1.32**. That function's own docstring records this having happened before.
3. ⚠️ **`variant` travels as a column, never in a filename.** `analyze_benchset_a.arm_of` maps filename
   substrings to arms and defaults everything unmatched to `"B1 supplied"`; 14 variant files would
   collapse into one arm, and `conflicting`/`keyword_flood` contain no matching substring at all.

Also: **`nykvist_evcharging` is not a SYNERGY collection.** Its `objective` is 160 characters against
~1,200 for every SYNERGY collection, so `raw_brief_map`'s "objective holds the review's abstract"
assumption fails there. Report it as an exception rather than averaging it in.

**`P-FB`.** The specificity bar as written is **unreadable, and would have been failed by data that
already exists**: 7 of 34 collections have `own < best_foreign`, against a bar of "≥6 of 34 flags →
report and stop". Because those 7 margins are already printed, any threshold chosen now is post-hoc.
So the probe splits in two, honestly labelled:

- The cosine-space margins are **descriptive and already measured** — cited, not re-registered.
- The **pre-registered blind test becomes a replication in lexical (BM25) space**, which has never
  been run as a brief × corpus matrix. `build_lexical_features(df, brief_map=…)` takes the seam, so it
  is free, and it is exactly the falsification pattern `CONTEXT.md` §7 mandates. **If a broken brief is
  a property of the brief, both representations flag it; if it is a property of one embedding's
  geometry, only one does.** That is a real test and its answer is not yet visible.

Two further corrections: validity is **n=8, not 34** (both validation CSVs cover only the 8 burned
`large_set_a` collections, and both are set-A-derived — so this is *not* independent validation and
must not be described as such), and `auc_floor` is **not** the §6 diagonal (qwen4b/weighted/set-A
held-out against jasper/whole-corpus; `synergy_moran_2021` is 0.488 against 0.436). Recompute one on
the other's population before joining.

**`P-CK`.** Two premises replaced. First, the **S-CE cue library citation is wrong** — `cues.py`
detects *novelty-introduction frames* for technology names, not predicates. The correct reuses are
`METRIC_HEADS`/`METRIC_RE` and `UNIT_RE` from the sibling repo's `discovery.py` and
`probe_metrics.py`; a comparator regex exists in **neither** repo and is new code. Second, the metric
**counts spans of text, not fields** — free-text fields sentence-split, list fields already one item
per span, with the benchset-28 substituting `population`/`interventions`/`outcomes`/`evidence.*_types`
(populated in all 28) for TIRI's always-empty-there criteria fields, because in review methodology
those *are* the eligibility framework. That keeps checkability a genuine `[0,1]` fraction, so the
IQR ≥ 0.2 language survives — pointed at a denominator that exists.

🔴 **Evidence availability must be query-conditioned or the metric is worthless.** A clinical abstract
carrying `95% CI`, `p<0.05`, `n=340` satisfies bare number+unit while stating no performance evidence
at all — and 26 of 28 benchset collections are clinical. A unit match counts only when it co-occurs
**in the same sentence** as a metric name from *that use case's own* vocabulary.

**`P-TX`.** The gate is replaced with one that measures the axis the probe names: **title-only against
title+abstract with the model held fixed**, fastembed on CPU, 2–3 small collections plus one TIRI
silo. Still $0, still ≥0.01 to proceed. The **`+venue` arm is dropped** — `papers_benchset_v1.parquet`
has no `venue` column, so it was unbuildable on its own confirmatory surface. Three things to fix
before any GPU spend if the gate passes: `text_variant` must enter the embedding cache **filename**
(`embed_benchsets.papers_path()` would otherwise let four variants overwrite each other, and the
resume logic would then skip a variant that never ran); `MAX_ABSTRACT_CHARS = 4000` truncates
abstracts but never titles, so `title_only` is the only untruncated arm and that is a confound between
arms; and the shuffled-brief seam **does not apply** here, because `P-TX` changes the paper side rather
than the brief side — stated explicitly so it does not read as an omission.

### Surface accounting, restated

§0 nominates `benchset_v1_large_set_b` as clean while `CONTEXT.md` §4 ring-fences "23 unused SYNERGY
reviews" as the only clean surface left. Those are not the same claim. **Decision: set B is spent, and
spent on `P-R` alone** — the one probe that needs realistic prevalence and enough positives to read a
win count. `P-FB` and `P-CK` run across all 34 by construction and consume no confirmatory surface.
Set B is burned after `P-R`, for one question rather than four. A strict read also excludes the 154
papers / 313 rows in `reports/benchset_v1_ab_crossing_papers.csv`.

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
| the ~~12~~ **13** collections too small to split (`benchset_v1_split_manifest.json` → `assignment`) | untouched | `P-FB` and `P-CK`, which need no train split |
| TIRI's own 6 use cases | burned for modelling, clean for *spec* questions | `P-CK` — they are the only analyst-written specs available |

> **Amendment:** §4 of `CONTEXT.md` ring-fences "**23 unused SYNERGY reviews**" as the only clean
> surface left, which is not the same claim as "set B is clean". Decision: **set B is spent, and spent
> on `P-R` alone.** `P-FB` and `P-CK` run across all 34 by construction and consume no confirmatory
> surface, so set B is burned for one question rather than four.

**Rule for all four probes: estimate on A, confirm on B, and report both.** A finding that
appears on A and not on B is a finding about A.

---

## `P-R` — The reader arm: do contradictions and fluff behave differently for an LLM?

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

The existing harness, unchanged: ~~`scripts/llm_pipeline_utils.py` `build_use_case_brief` is already
config-driven on `use_case_brief_cols`~~, and every response is disk-cached by (model, variant,
prompt version, params, prompt text). Reuse `run_spec_quality_ablation.variant()` to produce the
brief frames — **one function, both arms, so the two results are about the same 14 texts.**

> ⛔ **Amendment: right conclusion, wrong seam.** `build_use_case_brief` is the **layer-1 notebook**
> function; `use_case_brief_cols` appears nowhere in `scripts/`. The screening harness renders through
> **`render_brief`**, which is hardcoded to `BRIEF_FIELDS` and pinned by `BRIEF_RENDER_VERSION`. The
> reuse still works, but because `variant()` rewrites the **columns** `render_brief` reads — not
> because a config list is being passed. This matters for the instrument check below: the failure mode
> to guard against is not "a config column name is wrong" but "a variant rewrote a column
> `render_brief` does not read", which produces the same silent no-op.
>
> Two further consequences. **`run_llm_screening.py` cannot be used** — its `--corpus` and `--brief`
> are closed `choices` lists with no spec-variant hook and no set-B corpus, so this needs its own
> script importing `score_frame` directly. And **there is no few-shot path in the harness at all**;
> every prompt variant is zero-shot by construction, so "all 14, zero-shot" needs no switch.

- **Model:** `gemma-4-31b`, prompt **P2**. Not because it is best — it is the *weakest* of the four
  on set A — but because it is the model the whole B0/B1/B2 ladder ran on, so this joins that
  ladder rather than starting a new one. State that it is one family (caveat 3 of that report).
- **Variants:** all 14, zero-shot, so the brief is the only thing that varies.
- **Rows:** the same case-control sample and the same held-out rows as the matcher arm.
- **Metrics:** ROC-AUC **and** F2@own + fraction-read. The operating point is the thing a reader
  produces and a ranker cannot (`wf_llm_benchset_a_findings.md` §5), so a variant that holds AUC
  while collapsing recall is a failure this probe must be able to see.
- 🔴 **Added by amendment — the matcher arm must be re-run on set B, and it is free.** H1 and H3 are
  comparisons *between arms*, but the matcher's −0.014 and −0.072 were measured on set **A**'s 8
  collections. Reading a reader delta on B against a matcher delta on A would be the
  selection-on-holdout error in miniature: two surfaces, one subtraction. `run_spec_quality_ablation.py`
  gains a `--set` flag and the matcher re-runs on B's 7 collections for **$0** and a few minutes.
  Without this, H1 and H3 are not testable as written.

### Bars, fixed now

> ⛔ **Superseded by the Amendment.** Set B is **7 collections**, so "≥5 of 8" is unreadable there, and
> the matcher deltas quoted below were measured on set **A**. Corrected bars: `conflicting` costs the
> reader ≥0.03 more than it costs the matcher **on the same 7 collections**, on **≥5 of 7**; `H2`
> unchanged but reported with a win count beside the mean; `keyword_flood` costs the reader less than
> **the matcher's set-B delta**, on ≥5 of 7. Original text kept below.

| | Pass | ~~Kill~~ FAIL |
|---|---|---|
| **H1** | ~~`conflicting` costs a reader **≥0.03 more AUC than it cost the matcher**, on ≥5 of 8 collections~~ | `conflicting` is within 0.03 of the matcher's −0.014 → contradictions are cheap for everyone, and D43's three-critic linter collapses to two |
| **H2** | `fluff_replace` / `vague_objective` cost the reader **≥0.03** | inside the floor → prose damage is genuinely cold-start-only and the linter only needs to fire before the first labels |
| **H3** | ~~`keyword_flood` costs the reader **less** than the matcher's −0.072, on ≥5 of 8~~ | it costs *more* → term-list damage is universal, which simplifies the guidance and is worth knowing |

### Cost

9,993 rows was $0.42 for `gpt-oss-20b` and ~$2 for gemma across a cell. **14 variants is too many
to buy at full width.** Stage it:

1. **Stage 1 (~~~$1.50~~ ≈$0.80):** the 4 decisive variants — `full`, `conflicting`, `fluff_replace`,
   `keyword_flood` — on a 2,000-row subsample of set A. Estimates only, no verdict.
2. **Stage 2 (~~~$3.50~~ ≈$3.30):** the same 4 on **set B**, full held-out rows. **The verdict is read here.**
3. ~~Remaining 10 variants only if stage 2 shows the arms diverge at all.~~ **Out of budget** — see below.

**Ceiling $10.** Stop and report if stage 1 shows all four variants inside 0.03 of each other —
that is itself the answer (a reader is insensitive to spec form) and buying stage 2 adds nothing.

> ⛔ **Amendment: the cost model was ~2× optimistic, and the fix improves the measurement.** The
> measured gemma unit cost, read off the 64,857-record cache rather than estimated, is
> **$0.000132–0.000151/row**. Set B's case-control sample is **11,011 rows**, so 4 variants at full
> width is ≈**$6.2** and all 14 is ≈**$21.6** — over the ceiling.
>
> **Stage 2 therefore scores held-out rows only** (`split != "train"`, ≈4,400 rows). That is not just
> the cheap choice: it is the population the matcher arm scored, so it is the *comparable* one. Four
> variants plus the shuffled floor ≈ **$3.30**.
>
> **Stage 1 is cheaper than budgeted for a good reason:** gemma's set-A `P2` own-brief cell is already
> fully cached, so `full` is a 100% cache hit at **$0.00** and only the 3 degraded variants cost
> anything (≈$0.80). This is also the tripwire — if `full` reports fresh spend, the cache key has
> changed and stage 2 will cost double. **Stop if that happens.**
>
> **Revised total ≈$4.10 against a $10 ceiling.** The remaining 10 variants would add ≈$6.6 on B's
> held-out rows and are **explicitly out of budget** unless the ceiling is raised — which is a decision
> for after stage 2, not a silent overrun.

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

## `P-FB` — The foreign-brief detector: a free "your brief is broken" signal

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

- ~~Compute the full matrix on **all 34 use cases** (6 TIRI + 28 benchset), not the diversity
  notebook's subset.~~ **§6 already ran all 34** — `brief_transfer_auc` filters on the model's
  surfaces and jasper spans both, so `T` is already 34×34; only the *printed table* was `head(8)`.
  What this probe adds is validation, not coverage. It should, however, **persist the matrix**: it is
  saved nowhere and is recomputed on every notebook run. `usecase_diversity_utils` already has the loader.
- The statistic is the **margin**: `own − best_foreign`. Negative means broken.
- **Validation, and this is the part that makes it a probe rather than a plot:** the margin must
  predict something measured independently. Two candidates, both already on disk —
  (a) the collection's gain from labelling (`wf_label_budget_shape_grid.csv`), and
  (b) the collection's induced-brief improvement (`wf_llm_benchset_a_induced_features.md` — the
  machine-readable form is `wf_llm_benchset_a_induced_cosine.csv`), since a brief with room to improve
  is a brief that was weak.

> ⛔ **Amendment: this validation is n=8, on the burned surface, and is not independent.** Both CSVs
> cover only the 8 `large_set_a` collections, not 34 — so a `|rho| ≥ 0.5` bar needs a bootstrap CI
> beside it, and the result is **exploratory**. Both quantities are also set-A-derived, so validating a
> set-A margin against a set-A improvement is not the independent confirmation the word "independently"
> promises. It must not be described as such.
>
> One instrument mismatch to fix before joining: `auc_floor` in the grid CSV is
> `roc_auc_score(y, cos_brief_qwen4b)` on **weighted set-A held-out rows**, while §6's diagonal is
> **jasper over the whole collection**. They are not the same number — `synergy_moran_2021` is 0.488
> against 0.436. Recompute the own-brief AUC on `auc_gain`'s population, or the join silently compares
> two instruments. Also note the key formats differ: the grid CSV uses `synergy_moran_2021`, the
> induced CSV uses `moran_2021`.

### Bars, fixed now

> ⛔ **Superseded by the Amendment.** The specificity bar is unreadable: **7 of 34** collections
> already have `own < best_foreign`, against a bar of "≥6 of 34 → stop", and those 7 margins are
> already printed in the notebook, so no threshold chosen now is blind. Corrected bars below the
> original.

| | Pass | ~~Kill~~ FAIL |
|---|---|---|
| ~~**Specificity — the one that matters**~~ | ~~**≤2 of 34** use cases flag as broken. A detector that fires on healthy briefs trains the analyst to dismiss it — S-FR's own conclusion, and S-FF's generic-vocabulary failure is exactly how such a list fills up~~ | ~~≥6 of 34 flag → it is measuring corpus difficulty, not brief quality. Report and stop~~ |
| **Validity** | Spearman \|rho\| ≥ 0.5 against **at least one** of (a) or (b), and the sign is the predicted one | no relationship → the margin is a curiosity |
| **Confound control** | The margin must survive controlling for **prevalence**, which correlated with LOGO at rho −0.72 in the diversity notebook and is the obvious confound | it does not survive → report as confounded, do not ship |

**Corrected bars, fixed now:**

| | Pass | FAIL |
|---|---|---|
| **Cross-instrument specificity** — *the blind test, because lexical space has never been run* | Of the collections flagged at `margin < −0.03` in cosine space (the §5 floor), **≥half also flag in lexical space**, and **≤2** flag in lexical that cosine called healthy | The two representations disagree → the margin is a property of one embedding's geometry, not of the brief |
| **Validity** — *exploratory, **n=8** not 34, bootstrap CI required* | Spearman **rho ≤ −0.5** against ≥1 of (a) or (b). The sign is **negative**: a weak brief is one with more to gain from labelling | No relationship → the margin is a curiosity |
| **Confound control** | `partial_rho` given prevalence **keeps its sign and ≥half its magnitude** | It does not → report as confounded, do not ship |
| **Provenance stratification** | TIRI-6 (analyst-written briefs) and benchset-28 (review-abstract-derived) reported as two figures with two `n`s | `DATA_BRIEF.md` honest-limit #2: deriving a brief from the review's own abstract inflates *every* brief-reading score. Averaging the halves would compare two brief-generation processes |

**Specificity in cosine space is post-hoc and is labelled so.** It is reported as descriptive,
already-measured, and cited to `wf_usecase_diversity.ipynb` §6 — not re-registered as a prediction.

### Cost

**$0.** All vectors and briefs are cached. ~10 minutes of compute.

### What it changes if it passes

> **Amendment:** this caption is **already registered** as `NUMBERS.md` **N33** at status `LIVE+SQL`,
> quoting the `soil_microbiome` 0.603/0.792 pair. So `P-FB` is not proposing a new number — it is
> supplying the validation a `LIVE+SQL` row is currently shipping without. That raises the stakes on
> specificity rather than lowering them: a registered caption that fires on healthy briefs is worse
> than an unregistered one.

`academic_agent` gains a **⑤ Rebuild** caption with an owner: *"three other use cases screen your
papers better than your own brief does — your objective may be describing your intent rather than
your literature."* That is `NUMBERS.md`'s rule satisfied — a number with a computation behind it.
It is also the only brief-quality signal that needs **no labels at all**, which makes it the only
one available at the moment the brief is written.

---

## `P-CK` — Checkability: is this use case automatable, and can we say so before scoring?

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

- **Deterministic, no LLM.** Regex for number+unit patterns, comparators, and ~~the S-CE cue
  library~~ — **mis-cited: `cues.py` detects novelty-introduction frames for technology names, not
  predicates.** The correct reuses are `METRIC_HEADS`/`METRIC_RE` and `UNIT_RE` from the sibling
  repo's `discovery.py` and `probe_metrics.py`; a comparator regex exists in neither repo.
- Report both figures per use case across all 34, with `n` beside each.
- Cross the two: a use case with high checkability and low evidence availability is the
  **diagnosable failure** — the spec is fine and the corpus cannot answer it.
- **Pre-registered as DESCRIPTIVE.** No correlation against AUC is claimed at n=34 with the
  heterogeneity `CONTEXT.md` §5 documents. The deliverable is the figure and the reasoning.

### Bar

> ⛔ **Superseded by the Amendment.** "Fraction of the spec that is a predicate" has no denominator:
> `performance_criteria` is populated in 3 of 34 use cases and `decision_criteria` in 5 of 34. The
> metric is redefined to count **spans of text**, which keeps it a genuine `[0,1]` fraction and so
> keeps the IQR ≥ 0.2 language — pointed at a denominator that exists.

| Pass | ~~Kill~~ FAIL |
|---|---|
| ~~Both figures **vary materially across the 34** (IQR spanning ≥0.2) and the ordering is stable across two extraction rule sets~~ | the figures do not discriminate → the metric does not ship, exactly as S-UCQ Q5 pre-registers |

**Corrected bars, fixed now:**

| | Pass | FAIL |
|---|---|---|
| **Checkability spread** | Pooled IQR ≥ 0.2 over 34, **and** it survives jackknifing one use case at a time (§5: never decide from a single held-out use case) | Fails; **or** it passes pooled and vanishes within the TIRI-6 stratum → the spread is an artefact of two schemas, and the report says so |
| **Face validity** | `cement_binders` in the **top 2** of the TIRI 6 and `tech_forecasting` in the **bottom 2** | The one place this metric can be checked against something already known — S-AL: cement is AUC 1.000 under every model on checkable predicates; tech-intel has no performance criteria and is the hardest use case in the repo |
| **Rule-set stability** | Evidence availability: Spearman rho ≥ 0.9 **and** top-10 overlap ≥ 8/10 between `_UNIT_RE` and `UNIT_RE`. Checkability: rho ≥ 0.8 **and** ≥ 7/10 between the narrow and broad judgement lexicons | rho < 0.7 / < 0.6 respectively |

**Evidence availability is query-conditioned, not bare number+unit.** A clinical abstract carrying
`95% CI`, `p<0.05`, `n=340` satisfies bare number+unit while stating no performance evidence at all,
and 26 of 28 benchset collections are clinical. A unit match counts only when it co-occurs **in the
same sentence** as a metric name from *that use case's own* vocabulary. Without this the metric
measures "is this a clinical trial abstract", which is not the question.

### Cost

**$0**, no LLM, ~20 minutes.

### What it changes

A pre-scoring caption the funnel can show at ① or ⑤: *"your criteria are mostly judgement calls,
and 6% of these abstracts state a number — expect to label, not to automate."* It also gives
`NUMBERS.md` **N23** (criteria fields populated, currently 0 of 7 live use cases) a companion that
measures *quality* rather than presence.

---

## `P-TX` — What text the embedding sees

### The question

`CONTEXT.md` §6's rejected list is a graveyard of **scalar** feature engineering: 11 metadata
features (+0.002), PCA, per-use-case centring, four third branches, the induced brief in the
ensemble (+0.001). All of them add columns *beside* the embedding. **None of them changed what text
goes into it** — and `embedding_utils.MODEL_CONFIGS` already carries `title_abstract_sep` as a
per-model quirk, so the seam exists and has never been treated as a variable.

> **H6.** The paper text fed to the embedder is an unexplored lever, and it is not the same
> question as adding features — it changes the representation rather than annotating it.

Variants: `title_only` · `title+abstract` (shipped) · ~~`title+abstract+venue`~~ (**dropped — there is
no `venue` column on `papers_benchset_v1.parquet`, so the arm is unbuildable on its own confirmatory
surface; only the 2,873-row live corpus has one**) · **`cue_sentences`**
(the ~3 solution-claim sentences S-CE's cue library extracts at 85.5–90.6% coverage — free,
deterministic, and orthogonal to embedding similarity).

### The gate — this probe is expensive and must be earned

**Do not run `P-TX` until a $0 pre-check passes.** Re-embedding a corpus is Modal GPU time, and
`CONTEXT.md` §7 says route to Modal for genuinely heavy jobs, not on principle.

~~**Pre-check ($0):** the shipped export already carries **title-only** and **title+abstract**
material for the `cos_briefpre_*` / `cos_brief_*` column pairs on set A. If those two differ by
**<0.01 AUC**, the text-input axis is flat on the one comparison already paid for, and P4 is not
worth GPU time. **≥0.01 → proceed.**~~

⛔ **Wrong axis, and it would have produced a false FAIL.** `cos_briefpre_*` is
`embed_benchsets.BRIEF_VARIANTS["pre_screening"]` — it drops the review abstract from the **brief**
and adds the term lists. The **paper vectors are byte-identical** between that pair, so it says
nothing about paper text. It has also already been run: `03_eda_full_benchset_v1.ipynb` §8.4 measured
a paired median of 0.003–0.007 across 27 collections, below the 0.01 gate. And there is **no
title-only paper vector anywhere on disk** — `embed_benchsets.py` always calls `embed_papers`, which
always joins title+abstract.

**Corrected pre-check ($0):** embed **title-only** and **title+abstract** for 2–3 small collections
plus one TIRI silo with a **single fastembed model on CPU** — the model is held fixed, so the
contrast is genuinely about text input and nothing else. Compare within-silo AUC of cosine-to-brief.
**≥0.01 → proceed to Modal. Below → `P-TX` is closed and `CONTEXT.md` §6 gains a row.**

### Bars, fixed now

| Pass | ~~Kill~~ FAIL |
|---|---|
| Any variant beats `title+abstract` by **≥0.03 within-silo AUC on ≥5 of ~~8~~ 7 collections on set B** | all variants within 0.03 → the text input is not a lever, and the "features are already in the embedding" conclusion is complete rather than merely unrefuted |

> ⛔ **Amendment:** set B is **7 collections**. Note also that `MAX_ABSTRACT_CHARS = 4000` truncates
> abstracts but never titles, so `title_only` is the only arm with no truncation at all — a confound
> between arms, not merely a text change. And `text_variant` must enter the embedding cache
> **filename** before any GPU spend: `embed_benchsets.papers_path()` would otherwise let four variants
> overwrite each other, after which the resume logic silently *skips* a variant that never ran.

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
P-FB ($0) ─┬─► pass → ship a brief-quality caption with no labels required
           └─► fail → the only label-free brief signal is gone; P-R becomes the whole linter

P-CK ($0) ─┬─► pass → a pre-scoring "expect to label, not automate" caption
           └─► fail → drop, exactly as S-UCQ Q5 pre-registers

P-R ($4.10) ─┬─► arms diverge → D43's two-critic linter is correct and buildable
             └─► arms agree  → one linter, one verdict, and the guidance simplifies

P-TX (gate $0, then ~$3) ─┬─► pass → the first new feature lever since the query-conditioned block
                          └─► fail → close the feature-engineering question for good
```

**The honest expectation, recorded so it can be wrong:** `P-FB` passes on validity and **fails on
specificity** (detectors of this shape usually do); `P-CK` passes descriptively; `P-R`'s H1 passes and
H2 fails; `P-TX`'s pre-check kills it. If that is how it lands, the useful output is a two-critic
linter plus a checkability caption, and the feature question closes — which is worth the ~$4 to know.

> **Amendment note on that expectation.** `P-FB`'s specificity failure is **no longer a prediction** —
> 7 of 34 collections already flag, so it has already happened, and the honest reading is that the
> cosine margin as originally defined *is* partly measuring corpus difficulty. What remains genuinely
> open is whether the **conjunction** of cosine and lexical space is specific enough to ship. That is
> the question the amended probe asks.

## Outcome — 2026-08-14, all four run

Recorded here so this file reads as a closed loop rather than an open promise. Findings live in the
four reports; this is only the scoreboard against what was fixed above.

| probe | verdict | spend |
|---|---|---|
| **`P-FB`** | Cross-instrument specificity **FAIL** — 8 lexical-only flags against a bar of ≤2, and the two representations correlate at rho 0.33. Validity **PASS on set A, then FAIL on set B**: rho −0.929 became −0.464 with a CI of −1.000 to +0.765 and a sign that flips to +0.143 under prevalence control — and was never embedding-independent even on A (qwen4b −0.571, CI crossing zero). **Nothing ships; N33 is removed, not rewritten.** → [`wf_foreign_brief_detector.md`](wf_foreign_brief_detector.md), [`wf_foreign_brief_validity_setb.md`](wf_foreign_brief_validity_setb.md) | $0 |
| **`P-CK`** | Spread **FAIL** on both figures (IQR 0.055 and 0.179 against 0.2). Face validity **FAIL**. Stability **PASS** for evidence, **VACUOUS** for checkability. → [`wf_checkability_audit.md`](wf_checkability_audit.md) | $0 |
| **`P-R`** | H1 **FAIL**, H2 **FAIL** both halves, H3 **PASS**. Stop rule fired; stage 2 unbought. → [`wf_spec_quality_reader.md`](wf_spec_quality_reader.md) | $1.32 |
| **`P-TX`** | Gate **PASS** on 4 of 4, three of them clearing the 0.03 floor, **with inconsistent sign**. Modal run earned but not yet run. → [`wf_text_input_precheck.md`](wf_text_input_precheck.md) | $0 |

**Total $1.32 of a $10 ceiling.** `benchset_v1_large_set_b` was *not* spent on the reader arm (the stop
rule fired), and was instead spent on the one confirmatory question worth it: whether `P-FB`'s
surviving result survives a clean surface. It does not. That read is now used up.

### The honest expectation, scored

§Sequencing above predicted: *"`P-FB` passes validity and fails specificity; `P-CK` passes
descriptively; `P-R`'s H1 passes and H2 fails; `P-TX`'s pre-check kills it."* **Two of four.**

- ✅ **`P-FB` — right about specificity**, including which bar would fail. But the prediction said it
  would *pass* validity, and on the clean surface it does not. The plan was right that this detector
  fires too often and wrong that anything underneath it was salvageable.
- ❌ **`P-CK` did not pass descriptively.** It failed every bar it could fail. The prediction assumed
  the two figures would at least *vary*; the reason they do not is that half the metric's definition
  never fired, which no amount of forethought about the bars would have caught — only running it did.
- ❌ **`P-R`'s H1 failed rather than passed**, and it failed in the opposite direction: the reader did
  marginally *better* with a self-contradicting brief. H2 failed as predicted, but for the wrong
  reason — prose damage is not cold-start-only, it is *matcher-only*.
- ❌ **`P-TX` was not killed by its pre-check.** It is the only probe of the four that earned its next
  step — and only because the pre-check itself had to be rebuilt, since the one specified here
  measured the wrong axis and would have produced the predicted kill for the wrong reason.

That last line is the argument for the Amendment. **Three of these four probes would have produced a
publishable-looking number that was wrong**, and in `P-TX`'s case the wrong number would have
*confirmed the expectation written above* — which is exactly how a plan validates itself instead of
testing anything.

## What this plan does *not* cover

- **The label-noise ceiling** (24 unfilled adjudications in the sibling repo). It bounds every AUC
  in both repos including all four probes above, and it is human work, not a probe.
- **Prevalence-honest interface copy.** A design gap, not a measurement — no bar would be
  meaningful.

Both are detailed in the working notes rather than planned here, because neither is a probe.
