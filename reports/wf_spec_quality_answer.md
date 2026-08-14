# What makes a good use-case spec — the answer, and whether it can be scored

Status: **answer, 2026-08-14.** Consolidates `wf_spec_quality_ablation.md` (set A),
`wf_spec_quality_ablation_set_b.md` (clean), `wf_spec_quality_reader.md`,
`wf_foreign_brief_detector.md`, `wf_foreign_brief_validity_setb.md`, `wf_checkability_audit.md`,
`wf_text_input_precheck.md`. Every bar was pre-registered in `wf_spec_quality_plan.md`.

**The two questions this answers, and the honest headline for each:**

1. *What defines a high-quality vs low-quality spec, and what is the Minimum Viable / Optimal spec?*
   → **Answered, on two independent surfaces.** Three fields carry essentially all the value, three
   are decoration, and the ranking of failure modes is now priced. §2–§4.
2. *Can we score a use case, so the user knows where to improve and when it is done?*
   → **No single score. Yes, a priced linter plus a real completion test.** Three separate attempts at
   a single quality number were built and all three failed. §5–§7. **The reason they failed is itself
   the answer**, and it is not a gap in the work.

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

**Surfaces.** Set A = 8 collections, burned (five prior selection passes). Set B = 7 collections,
**never scored by anything before this week**. Numbers are quoted `A / B`. Where they disagree, B wins.

⚠️ **Deleting a field pins its feature to exactly 0.500** — the feature becomes uncomputable and
degenerates to a coin flip. So a deletion's "delta" is `full − 0.500`, i.e. **what the field was
worth**, not how much damage was done. Read those rows that way.

⚠️ **The biggest limit, stated up front.** All 15 collections measured are SYNERGY/benchset —
**biomedical, clinical and social-science reviews**. TIRI's own six use cases are
technology/hard-science/industry (`CONTEXT.md` §§4–5, 49–51). Whether any of this transfers to a
cement or a satellite use case is **unmeasured**, and it is the single largest open risk in this answer.

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

## 7. What is still unanswered

1. 🔴 **Transfer to hard science.** All 15 collections are biomedical/clinical/social-science. TIRI's
   six use cases are technology and industry. **Nothing here has been measured on the population it
   would ship to.** This is the largest open risk in the answer.
2. 🔴 **Interactions.** Every variant changes **one** field. Whether a strong objective rescues a weak
   term list — the question a person filling in a form actually has — is untested. `only_name` is the
   only multi-field degradation measured.
3. 🟡 **The reader arm is set A only, one model family** (`gemma-4-31b-it`, the weakest of four on
   set A). A stronger reader could be more sensitive to spec form, not less.
4. 🟡 **Prose contradiction for a reader.** S-AL measured a *prose* contradiction costing a reader
   0.828 → 0.772. The variant here contradicts the *term lists* and costs +0.016. **S-AL is
   un-addressed, not refuted**, and closing it is a ~$0.30 experiment.
5. ⚪ **Whether telling someone to write better prose works.** S-FF's grader gap (+0.27 recall@10%
   between an oracle and a failing human grader) is explicitly unresolvable in simulation. It needs one
   user session.
6. ⚪ **The label-noise ceiling** — 24 unfilled adjudications in the sibling repo bound every AUC
   above, including all of these.
