# What makes a good use-case spec — the answer, and whether it can be scored

Status: **answer, 2026-08-14.** Consolidates `wf_spec_quality_ablation.md` (set A),
`wf_spec_quality_ablation_set_b.md` (clean), `wf_spec_quality_reader.md`,
`wf_foreign_brief_detector.md`, `wf_foreign_brief_validity_setb.md`, `wf_checkability_audit.md`,
`wf_text_input_precheck.md`. Every bar was pre-registered in `wf_spec_quality_plan.md`.

**The two questions this answers, and the honest headline for each:**

1. *What defines a high-quality vs low-quality spec, and what is the Minimum Viable / Optimal spec?*
   → **Answered, on three independent surfaces including TIRI's own hard-science use cases.** Three
   fields carry essentially all the value, three are decoration, and the ranking of failure modes is
   priced and transfers. §2–§4, confirmed in §8.1.
2. *Can we score a use case, so the user knows where to improve and when it is done?*
   → **No single score. Yes, a priced linter plus a real completion test.** Three separate attempts at
   a single quality number were built and all three failed. §5–§6. **The reason they failed is itself
   the answer**, and it is not a gap in the work.
3. *And the question a form-filler actually asks — does one good field cover for a bad one?*
   → **No. Defects compound rather than rescue**, in 6 of 6 cells across two surfaces. §8.2.

---

## 1. How to read any number here

**Every score is ROC-AUC: the chance a genuinely relevant paper is ranked above an irrelevant one.
0.500 is a coin flip.** A gap under **0.03** is *not established* (`CONTEXT.md` §5: seed-to-seed noise
is ~0.010).

**Two consumers, two label regimes, four cells — and this is why there is no single answer.**

|  | **0 labels (cold start)** | **after ~60 labels** |
|---|---|---|
| **matcher** (BM25 / term overlap) | reads the brief's prose *directly*, nothing can compensate | can reweight around a bad field |
| **reader** (LLM screening one paper at a time) | reads the whole brief redundantly | n/a — zero-shot either way |

A defect that is catastrophic in one cell is free in another. Any advice that does not name the cell is
not advice.

**Surfaces.** Set A = 8 benchset collections, burned (five prior selection passes). Set B = 7 benchset
collections, **never scored by anything before this week**. **TIRI = the 6 live use cases**, the
population this guidance ships to. Numbers in §2–§4 are quoted `A / B`; §8 adds TIRI. Where A and B
disagree, B wins; where benchset and TIRI disagree, **say both** — they are different populations, not
a better and a worse measurement of one.

⚠️ **Deleting a field pins its feature to exactly 0.500** — the feature becomes uncomputable and
degenerates to a coin flip. So a deletion's "delta" is `full − 0.500`, i.e. **what the field was
worth**, not how much damage was done. Read those rows that way.

✅ **The limit that used to be listed here is now closed.** §2–§4 were measured on SYNERGY/benchset —
biomedical, clinical and social-science reviews — while TIRI's own six use cases are
technology/hard-science/industry (`CONTEXT.md` §§4–5, 49–51). **§8.1 now measures all 14 variants on
those six**, and the MVP table survives. One recommendation's magnitude does not transfer
(`must_only_one`); everything else does.

⚠️ **The limit that remains is prevalence.** The three surfaces run 2.19% / 1.87% / **57.6%** positive.
That span is *why* the field ranking holding across all three is meaningful — but no absolute number
here is a production estimate.

---

## 2. 🟢 What each field is actually worth

**At 0 labels — what the field buys, above a coin flip:**

| field | feature it feeds | worth (A) | worth (B) | verdict |
|---|---|---|---|---|
| **`objective`** | `bm25_obj` | **+0.151** | **+0.263** | 🟢 the single most load-bearing field, by a wide margin, on both surfaces |
| **`terms_nice_to_have`** | `bm25_nice` | **+0.105** | **+0.215** | 🟢 second-most, and **twice as valuable on clean data** |
| **`terms_must_include`** | `overlap_must_frac` | cutting to 1 term: −0.103 | −0.175 | 🟢 real, and it is *count* that matters, not presence |
| `problem_statement` | — | **0.000** | **0.000** | ⚪ decoration |
| `domain_*` | — | **0.000** | **0.000** | ⚪ decoration |
| `terms_exclude` | — | **0.000** | **0.000** | ⚪ decoration *as spec text* |

**After ~60 labels — `d` against the full brief, with collections worse than the 0.03 floor:**

| variant | A | B | verdict |
|---|---|---|---|
| `only_name` — the 2–4 word name alone | −0.105 (4/7) | −0.045 (3/7) | 🟢 **worst variant on both surfaces** |
| `keyword_flood` — term lists padded with generics | −0.072 (6/7) | −0.031 (3/7) | 🟡 halves; direction survives twice |
| `no_must` | −0.043 (4/7) | **−0.010 (1/7)** | 🔴 **does not replicate** |
| `must_only_one` | −0.012 (3/7) | −0.010 (1/7) | ⚪ inside the floor on both |
| `conflicting` — exclusions moved into must-include | −0.014 (1/7) | −0.009 (1/7) | 🟢 **cheap for a matcher, twice confirmed** |
| `no_obj` / `no_nice` / `no_prob` / `no_domain` / `no_exclude` | ≤0.004 | ≤0.007 | ⚪ all inside the floor |

**Hard finding — the two regimes invert.** Prose damage is *catastrophic* at cold start and **free**
after labels (`no_obj` costs 0.151–0.263 at zero labels and ≤0.007 fitted). Term damage is the mirror
image: **invisible** at cold start, load-bearing after labels. *"Is this spec good?"* has no answer
until you say **for which consumer, at what label count** — this is the most robust result in the body
of work and it replicates cleanly.

**Hard finding — three of the schema's fields are decoration to every consumer measured.**
`problem_statement`, `domain_*` and `terms_exclude` are worth **0.000 at cold start and ≤0.007 after
labels, on both surfaces**. Keep them for humans if they help someone think; **do not make them
required fields.** (`terms_exclude` as a *priced, reversible filter* is a real instrument — that is a
different mechanism from putting the words in the spec.)

---

## 3. 🟢 The four ways of writing it badly, ranked and priced

| how you wrote it badly | cold start (A / B) | after labels (A / B) | reader (A only) |
|---|---|---|---|
| **only the topic name** | −0.151 / **−0.263** | −0.105 / −0.045 | −0.038 |
| **vague one-liner objective** | −0.209 / **−0.194** | −0.002 / −0.002 | **+0.010** |
| **generic fluff replacing prose** | −0.177 / **−0.169** | +0.003 / −0.009 | **+0.003** |
| **term lists padded with generics** | −0.089 / −0.067 | −0.072 / −0.031 | **−0.007** |
| **self-contradicting term lists** | −0.007 / −0.028 | −0.014 / −0.009 | **+0.016** |
| *fluff **appended** to real prose* | −0.018 / −0.006 | +0.008 / −0.004 | — |
| *(reference) the wrong brief entirely* | — | — | **−0.282** |

**Hard finding — dilution is nearly free, replacement is not.** Appending waffle to a real objective
costs ~0.01 on both surfaces; *replacing* the objective with waffle costs 0.17. So the guidance is
**"say the real thing somewhere"**, not "be concise".

🔴 **One widely-quoted claim does NOT replicate.** On set A, fluff (0.473) and vagueness (0.441) drove
the objective feature **below chance** — worse than deleting the field (0.500). On set B they are
**0.594 and 0.569**, i.e. above chance and *better* than deletion. On clean data the ordering is
monotone: real prose > diluted > generic > vague > nothing. **Any prose beats no prose.** The practical
advice barely changes; the rhetorical claim must be dropped.

🟢 **A reader is robust to all of it.** Every localised degradation costs an LLM screener less than the
noise floor — three are *positive*. Stripping the brief to its bare topic name costs only −0.038, while
a *wrong* brief costs **−0.282**. So the mechanism is **redundancy**: no single field is load-bearing
for a reader because the topic is restated across several of them. That is a different claim from "spec
quality does not matter to a reader", and only the first is supported.

---

## 4. 🟢 The answer: Minimum Viable and Optimal spec

### Minimum Viable — omit any of these and there is measurable loss

| field | written how | evidence |
|---|---|---|
| **`objective`** | **2–4 sentences of real subject matter that read like the papers you want.** A document, not an instruction and not a summary of your intent. | Worth **+0.151 / +0.263** above chance at zero labels — the largest single-field effect measured anywhere in this work, on both surfaces. Replacing it with generic prose or a vague line costs ~0.17–0.21 of that. |
| **`terms_nice_to_have`** | **≥5 discriminative phrases.** | Worth **+0.105 / +0.215**. Second-largest, and **it doubled on clean data** — the most under-reported field in the schema. |
| **`terms_must_include`** | **5–8 precise phrases. Not 1, not padded.** | One term costs **−0.103 / −0.175** at cold start. Padding costs −0.072 / −0.031 after labels. Both ends are priced. |

### Optimal — measured to pay, beyond the minimum

| addition | evidence |
|---|---|
| **Term lists re-derived from labels at ~60**, never hand-written | **+0.091**, the largest single-feature gain measured in this repo |
| **2–4 pinned exemplars** (boundary keeps + near-misses) | Sibling repo: sensitivity **0.52 → 0.77**, the biggest lever measured there |
| **Exclusions as reversible, priced filters** — not as spec text | `terms_exclude` in the spec is worth 0.000; the same knowledge as a filter is a real instrument |

### Never

| do not | evidence |
|---|---|
| ship a 2–4 word topic name alone | **−0.151 / −0.263** cold, −0.105 / −0.045 fitted. Worst variant on both surfaces — **and it is what `embedding_utils.get_use_case_text` falls back to today** |
| pad term lists with generic vocabulary | −0.089 / −0.067 cold, −0.072 / −0.031 fitted. Worse than an empty field on both |
| write fluff or a vague objective | costs ~0.17–0.21 of the largest effect in the schema |
| use a per-criterion checklist prompt | `CONTEXT.md` §6: F2@own collapses to 0.24–0.50 |

### Optional — keep for humans, never require

`problem_statement`, `domain_*`. **0.000 to every consumer, on both surfaces.**

---

## 5. 🔴 Can we score a spec? Three attempts, three failures

A single quality number was attempted three times. All three are dead, and each died differently:

| attempt | idea | why it failed |
|---|---|---|
| **Checkability + evidence availability** (`P-CK`) | fraction of criteria a machine could test, × fraction of abstracts stating that evidence | **Does not discriminate.** Pooled IQR **0.055** against a pre-registered 0.2 bar. And it ranked `tech_forecasting` — no performance criteria, hardest use case in the repo — **3rd of 6** instead of last. Half its own definition never fired |
| **Foreign-brief margin** (`P-FB`) | if a stranger's brief screens your corpus better than your own, yours is broken | **An embedding artefact.** Rebuilt in BM25 space, **8 new collections flag that the embedding called healthy** (bar: ≤2), rho between representations **0.33**, and the motivating −0.190 becomes −0.021. Its fallback caption died too: rho −0.929 → **−0.464 on clean data, sign flipping to +0.143 under prevalence control** |
| **Criteria-populated count** (`N23`) | how many of the fields that pay are filled | **Measures presence, not quality.** Directly demonstrated: `ner` and `solar_leo` have populated `performance_criteria` whose targets are *"improvement over baseline"* and *"better than the incumbent"* — judgement invitations wearing a predicate's clothing. Only `cement_binders` carries a real numeric comparator, so "3 of 6 populated" overstates the true count threefold |

### 🟢 Why no single score is possible, which is the real answer

**A single number would have to average over regimes that point in opposite directions.** From §2:
prose damage is worth −0.26 at zero labels and −0.007 after sixty. Term damage is 0.000 at zero labels
and −0.031 after sixty. A spec that is excellent for a cold-start ranker and useless for a fitted one
would score "medium" — and *medium is wrong in both regimes*.

*ELI18: it is like asking for one number that says how good a car is. A car can be great for city
driving and terrible for towing. Averaging those gives you a number that misdescribes both, and
somebody buys the wrong car. The useful output is not one score — it is a short list of specific,
priced warnings plus a test for when you are done.*

---

## 6. 🟢 What *is* buildable: a priced linter, and a real completion test

### Before any labels — a deterministic linter. Every check is free; every cost is measured.

| check (all label-free, all deterministic) | how | measured cost if it fires | priority |
|---|---|---|---|
| `objective` empty | field check | **−0.151 / −0.263** — the feature dies | 🔴 blocker |
| only the name is populated | field check | worst variant on both surfaces | 🔴 blocker |
| `terms_nice_to_have` empty | field check | **−0.105 / −0.215** — the feature dies | 🔴 blocker |
| `terms_must_include` has 1 term | count | −0.103 / −0.175 | 🔴 blocker |
| `objective` is short and/or mostly generic vocabulary | word count + overlap with the S-FF generic list | vague keeps only +0.069 of +0.263 | 🟠 warn |
| term lists overlap the generic-vocabulary list | set intersection against S-FF's measured list (`used`, `using`, `analysis`, `data`, …) | −0.089 / −0.067 cold, −0.072 / −0.031 fitted | 🟠 warn |
| `terms_exclude` ∩ `terms_must_include` ≠ ∅ | set intersection | −0.014 / −0.009 matcher, **+0.016 reader** | 🟡 low — priced as near-free |
| `problem_statement` / `domain_*` empty | field check | **0.000** | ⚪ **do not warn** |

**Every row has a number behind it**, which satisfies `NUMBERS.md`'s standing rule. Note the last two
rows especially: the contradiction check is worth *building* but not worth *blocking on*, and warning
about empty decoration fields would train the analyst to dismiss the linter — which is S-FR's own
conclusion about detectors that cry wolf.

### After ~60 labels — the completion test, and it is validated

**The shuffled-brief control.** Rebuild the brief-reading features against a **deliberately wrong**
brief (a derangement — no use case keeps its own) and compare. `build_lexical_features(df,
brief_map=…)` already takes the seam.

**Why this one and not the others:** it is the only brief-quality instrument that has passed on every
surface it has been tried on. Lexically, AUC **0.777 → 0.498** with predicted-positive rate → **0.000**
on SYNERGY, and 5/6 in-repo. For an LLM reader, **−0.282 AUC / −0.297 F2@own / −15.8pp fraction-read**,
worse than the floor on 7 of 8 collections. It is the one thing in this whole body of work that
survived contact with every surface.

> **When is a spec done?** When the linter is silent, **and** — once you have ≥20 positives and ≥50
> negatives labelled (`wf_label_budget_shape.md`'s measured gate) — **your own brief beats a deranged
> brief on your own corpus.** If it does not, your brief is measuring generic paper quality rather than
> your question, and no amount of rewriting fields will fix that.

⚠️ **The honest asymmetry.** The linter is label-free but only catches *known* failure modes. The
completion test is general but **needs labels**, so it cannot run at the moment the brief is written.
There is no validated instrument that is both. That gap is real, and every attempt to close it (§5)
has failed.

---

## 8. 🟢 Transfer to hard science, and the interaction test — both now measured

§7 listed these as the two largest gaps. Both are now closed, on **TIRI's own six use cases** —
cement, carbon capture, satellites, soil microbiology, NER and technology forecasting. 1,848 rows,
57.6% positive. `$0`, no LLM.

### 8.1 What transfers — three surfaces, `cold` = zero-label worst-affected feature, `fit` = after 60 labels

| variant | A `cold`/`fit` (of 8) | B `cold`/`fit` (of 7) | **TIRI** `cold`/`fit` (of 6) | transfers? |
|---|---|---|---|---|
| `no_obj` | −0.151 / −0.003 | −0.263 / −0.007 | **−0.100 / −0.003** | 🟢 **yes** — load-bearing at cold start on all three |
| `no_nice` | −0.105 / −0.002 (0/8) | −0.215 / −0.005 (0/7) | **−0.118 / −0.016 (2/6)** | 🟢 **yes**, and on TIRI it is the *largest* cold-start effect |
| `only_name` | −0.151 / −0.105 (4/8) | −0.263 / −0.045 (3/7) | **−0.118 / −0.087 (3/6)** | 🟢 **yes** — worst single-field variant on all three |
| `keyword_flood` | −0.089 / −0.072 (6/8) | −0.067 / −0.031 (3/7) | **−0.074 / −0.063 (4/6)** | 🟢 **yes** — the most robust after-labels finding |
| `conflicting` | −0.007 / −0.014 (1/8) | −0.028 / −0.009 (1/7) | **0.000 / −0.001 (0/6)** | 🟢 **yes** — cheap for a matcher on all three |
| `no_prob` | 0.000 / −0.000 | 0.000 / −0.000 | **0.000 / −0.005 (1/6)** | 🟢 decoration on all three |
| `no_domain` | 0.000 / −0.002 | 0.000 / −0.001 | **0.000 / −0.009 (1/6)** | 🟢 decoration on all three |
| `no_exclude` | 0.000 / −0.004 | 0.000 / −0.005 | **0.000 / 0.000 (0/6)** | 🟢 decoration on all three |
| `no_must` | 0.000 / −0.043 (4/8) | 0.000 / −0.010 (1/7) | **0.000 / −0.032 (2/6)** | 🟡 unstable — 0.043 / 0.010 / 0.032 |
| `must_only_one` | −0.103 / −0.012 | −0.175 / −0.010 | **−0.039 / −0.003** | 🔴 **much weaker on TIRI** |

**Hard finding — §4's MVP table survives the transfer test.** All three MVP fields are load-bearing on
the hard-science corpus, all three "decoration" fields are still decoration, and the two headline
"never do this" items (`only_name`, `keyword_flood`) hold on all three surfaces. `keyword_flood` at
−0.063 on 4 of 6 is *stronger* on TIRI than on clean benchset data.

**One recommendation weakens.** `must_only_one` costs only −0.039 at cold start on TIRI against
−0.103/−0.175 on benchset. The likely reason is visible in the specs themselves: TIRI's
`terms_must_include` lists are already only **2–10 terms** (`solar_leo` has 2), so truncating to one
is a much smaller edit than truncating a benchset list of 7–8. So *"not 1 term"* holds; the specific
magnitude does not transfer.

⚠️ 🔴 **And the specs already fail the recommendation.** TIRI's own six carry `terms_must_include`
2–10 and `terms_nice_to_have` **1–6**, against §4's recommended 5–8 and ≥5. `objective` is 76–369
characters against ~1,200 on benchset. **TIRI's own use cases mostly fail TIRI's own advice**, which
is worth knowing before the advice becomes a form — and it means the linter in §6 would fire on them.

🟡 **The "below chance" claim reappears here, weakly.** On TIRI the objective feature is 0.600 with a
real objective, **0.499 with fluff and 0.491 with a vague one-liner** — i.e. at or just below a coin
flip, as on set A (0.473/0.441) and unlike set B (0.594/0.569). Two of three surfaces show it. Treat
it as **surface-dependent**, not as a rule: what is safe to say on all three is *"replacing the
objective with generic prose destroys essentially all of its value."*

⚠️ Read absolute AUC with care: TIRI runs **57.6% positive** against benchset's 1.9–2.2%, roughly 20×
production prevalence (`CONTEXT.md` §3). What transfers here is the **ranking of fields**, not the
levels. And `terms_exclude` is empty on 3 of 6, so `no_exclude`/`conflicting` measure 3 use cases.

### 8.2 🟢 The interaction: defects **compound**, they do not rescue

The question a form-filler actually has — *"my objective is good but my keywords are lazy, does that
matter?"* — needs a pair, not two singles. Three pairs, two surfaces, after 60 labels:

| pair | surface | defect A | defect B | A+B | **measured pair** | excess | worse than floor |
|---|---|---|---|---|---|---|---|
| `fluff_replace` + `keyword_flood` | TIRI | −0.015 | −0.063 | −0.078 | **−0.097** | **−0.020** | **5 of 6** |
| | B | −0.009 | −0.031 | −0.040 | **−0.060** | **−0.020** | **5 of 7** |
| `vague_objective` + `keyword_flood` | TIRI | −0.005 | −0.063 | −0.068 | −0.081 | −0.013 | 4 of 6 |
| | B | −0.002 | −0.031 | −0.033 | −0.042 | −0.009 | 4 of 7 |
| `fluff_replace` + `no_must` | TIRI | −0.015 | −0.032 | −0.047 | −0.055 | −0.008 | 4 of 6 |
| | B | −0.009 | −0.010 | −0.019 | −0.035 | −0.016 | 4 of 7 |

**Hard finding — there is no rescue effect. In 6 of 6 cells the pair is worse than the sum of its
parts**, and `fluff+flood`'s excess is **−0.020 on both surfaces independently**. On TIRI,
`fluff_and_flood` is worse than the floor on **5 of 6 use cases** — a higher win count than any
single-field variant including `only_name`.

*ELI18: we hoped a well-written description might cover for a lazy keyword list — that you could get
one half right and be fine. You cannot. Getting both halves wrong is worse than adding up the cost of
each mistake separately: the two defects make each other worse rather than one propping the other up.*

🟡 **The honest limit on the size.** Each individual excess (0.008–0.020) is **inside the 0.03 noise
floor**, so no single one is established on its own. The evidence is the **consistency** — same sign in
6 of 6 cells across two independent surfaces, with one value reproducing to three decimal places. Read
it as *"no evidence of rescue, and consistent weak evidence of compounding"*, not as "compounding is
worth 0.02".

**What it changes in §6's linter — two things:**

1. 🟢 **Block on each defect independently.** A strong `objective` does not license a lazy term list,
   and vice versa. The blockers in §6 stand as written.
2. 🟢 **Escalate when more than one fires.** Two warnings are worse than twice one warning. A spec
   tripping both a prose check and a term check is in disproportionately worse shape than the two
   numbers would suggest — which is the one thing a per-field checklist cannot express and is worth
   a sentence in the UI.

---

## 9. What is still unanswered

1. ✅ ~~**Transfer to hard science.**~~ **Closed — see §8.1.** Measured on TIRI's own six. The MVP
   table survives; `must_only_one`'s magnitude does not transfer.
2. ✅ ~~**Interactions.**~~ **Closed — see §8.2.** Three pairs on two surfaces: defects **compound**,
   they do not rescue, in 6 of 6 cells.
3. 🔴 **Prevalence.** TIRI's six run 26–77% positive, ~20× production. The *ranking* of fields now
   holds on three surfaces spanning 1.9% to 57.6% prevalence, which is real evidence of robustness —
   but no absolute number here is a production estimate.
4. 🟡 **The reader arm is set A only, one model family** (`gemma-4-31b-it`, the weakest of four on
   set A). A stronger reader could be more sensitive to spec form, not less.
5. 🟡 **Prose contradiction for a reader.** S-AL measured a *prose* contradiction costing a reader
   0.828 → 0.772. The variant here contradicts the *term lists* and costs +0.016. **S-AL is
   un-addressed, not refuted**, and closing it is a ~$0.30 experiment.
6. ⚪ **Whether telling someone to write better prose works.** S-FF's grader gap (+0.27 recall@10%
   between an oracle and a failing human grader) is explicitly unresolvable in simulation. It needs one
   user session.
7. ⚪ **The label-noise ceiling** — 24 unfilled adjudications in the sibling repo bound every AUC
   above, including all of these.
