# The reader arm — no single brief field is load-bearing for a reader, because the signal is redundant

Status: **stage 1 complete, stage 2 deliberately not bought.** The pre-registered stop rule fired.
H1 FAIL, H2 FAIL (both halves), H3 PASS. Total spend **$1.32** against a $10 ceiling.

🟢 clears the 0.03 noise floor / decisively measured · 🟡 real but under the floor · ⚪ engineering finding

**Answer, in one line:** a brief's prose is worth about **0.04 AUC** to an LLM screener in total
(stripping it to the bare topic name costs −0.038), but **no single way of writing it badly costs
anything** — contradiction, keyword padding, fluff and vagueness all land inside the noise floor,
because the topic signal is spread **redundantly** across the brief's fields and corrupting one
leaves the others carrying it. Replace the content outright and it costs **−0.282**.

So the reader is not insensitive to spec quality; it is **robust to localised spec damage**. Those
are different claims and only the second one is supported here.

Probe `P-R` of [`wf_spec_quality_plan.md`](wf_spec_quality_plan.md). **Read that file's Amendment
before any number here** — the bars are fixed there, and three of this plan's four probes had
premises that did not survive contact with the code.

---

## 0. How to read any number here

**Every score is ROC-AUC unless labelled otherwise:** the chance that the model ranks a genuinely
relevant paper above an irrelevant one. **0.500 is a coin flip.** `d_` columns are the change
against the full shipped brief, so **negative means the damaged brief did worse**.

**A gap under 0.03 is not a result.** Re-running with different seeds moves within-silo AUC by about
0.010 on its own, and `CONTEXT.md` §5 sets ~0.03 as the threshold below which two approaches are
*not established* as different. Almost everything in §2 is inside that band, and that is the finding
rather than a weakness of it.

**The floor is the most important row in the table.** A null result is uninterpretable on its own —
"the reader did not notice" and "our instrument cannot see anything" produce identical numbers. So
every table carries the **shuffled-brief derangement**: every collection scored against a *different*
collection's brief, no collection keeping its own. `CONTEXT.md` L258–260 requires this of anything
claiming to read a brief. If the degradations sit at zero *and* the derangement sits at zero, the
instrument is blind and nothing here means anything. It does not.

**Two matcher comparators, both printed, because neither is a clean match.** H1 and H3 ask whether a
reader is hurt *more or less than a matcher*, which needs a matcher number to subtract. The matcher
produced two — a **fitted** arm at 60 labels (the comparator the plan named) and an **unfitted**
single-feature arm at 0 labels (the one actually label-matched to a zero-shot reader). The unfitted
one has a trap: each variant damages a *different* feature, so reading one fixed column would report
a confident 0.000 for variants it simply does not measure. The comparator below is therefore the
**most negative** of the three unfitted features — "the most the matcher noticed at zero labels".

**Label which critic is speaking.** The AUC, F2 and win counts are hard numbers off held-out rows.
Which comparator is the *right* one, and what the null in §2 means, are judgement calls and are
marked.

**Surface and instrument.** `benchset_v1_large_set_a`, a seeded 1,998-row subsample of the 9,993-row
case-control sample, 8 collections, 271 positives. `google/gemma-4-31b-it`, prompt `P2`, zero-shot —
the same model and prompt the whole B0/B1/B2 ladder ran on, chosen because it joins that ladder
rather than starting a new one, *not* because it is the best of the four (it is the weakest on set A).
**One model family, so this is unreplicated.** Parse rate **100.0%** on every cell.

⚠️ **Tie fraction is 0.94–1.00.** Verbalised 0–100 scores are extremely lumpy — the model emits a
handful of distinct values — so AUC here is a coarse instrument by construction. That is a real limit
on resolving small differences, and it is why the derangement floor matters so much: it demonstrates
the instrument can move 0.28 when the brief is genuinely wrong.

**Instrument check, run before any verdict.** Each variant's rendered brief was hashed per
collection and asserted distinct: **6 variants, 6 distinct texts on every collection.** This guards
the failure the plan mis-described. The plan said the seam was `build_use_case_brief`'s
`use_case_brief_cols` config; it is not — that is a layer-1 notebook function no script calls, and
the harness renders through `render_brief`, hardcoded to `BRIEF_FIELDS`. So the risk is not a wrong
config key but **a variant rewriting a column `render_brief` never reads**, which renders identically
to `full`, scores identically, and reads as "this degradation is free". The assert raises rather than
warns, because a silent no-op is indistinguishable from a real null.

---

## 1. 🟢 The floor: the instrument works, and it is not subtle

| cell | d AUC | d F2@own | d fraction-read | collections worse than the 0.03 floor |
|---|---|---|---|---|
| **shuffled** (derangement) | **−0.282** | **−0.297** | **−0.158** | **7 of 8** |

Give each collection somebody else's brief and the reader loses 0.282 AUC, its F2 at its own
operating point collapses by 0.297, and it stops flagging papers at all — reading 15.8 percentage
points less of the corpus. This reproduces the published set-A floor (0.777 → 0.498, F2 0.285 →
0.000) on a subsample, and that cell was already paid for, so it cost **$0.00**.

*ELI18: we deliberately handed the model the wrong instructions — the brief for a completely
different research question — and its ability to spot relevant papers fell off a cliff. That is the
control that makes the rest of this report readable: it proves the model really is reading the brief,
so when the damaged briefs below change nothing, that is about the damage, not about a broken test.*

⚠️ The one exception is `moran_2021`, where the derangement scores **+0.068**. Its own-brief AUC is
**0.455 — below a coin flip**. A brief cannot be made much worse than actively wrong, and
`CONTEXT.md` already records `moran_2021` as a collection whose inclusion rule is learnable in
representation space and *not statable as a rule*. It is excluded from nothing, but read it as a
collection where the brief was never working rather than as a case of a wrong brief helping.

---

## 2. The four degradations: all inside the floor, scattering both directions

| variant (what it imitates) | reader d AUC | reader d F2@own | matcher d, fitted | matcher d, unfitted worst | reader worse than floor | reader **better** by >0.03 |
|---|---|---|---|---|---|---|
| `conflicting` — exclusions moved into must-include | **+0.016** | +0.041 | −0.014 | −0.014 | 1 of 8 | 2 of 8 |
| `vague_objective` — objective replaced by a one-liner | **+0.010** | +0.015 | −0.002 | −0.209 | 2 of 8 | 2 of 8 |
| `fluff_replace` — prose replaced by generic academic filler | **+0.003** | +0.004 | +0.003 | −0.177 | 2 of 8 | 1 of 8 |
| `keyword_flood` — must/nice padded with generic vocabulary | **−0.007** | +0.011 | −0.072 | −0.106 | 0 of 8 | 0 of 8 |
| *(reference)* `shuffled` | −0.282 | −0.297 | — | — | 7 of 8 | 0 of 8 |

The whole spread across the four is **0.023** — smaller than the 0.03 floor, and smaller than the
0.010 seed-to-seed noise multiplied across eight collections. Three of the four are *positive*.

**Per-collection, the signature is noise rather than effect:** `conflicting` is worse than the floor
on 1 collection and *better* by more than the floor on 2. `vague_objective` is 2 and 2. A real effect
does not scatter symmetrically.

### 🟢 2a. The diagnostic that explains the null — and it is redundancy, not indifference

Every variant above leaves `use_case_name` intact (`variant()` never clears it), so a reader that
reconstructs the topic from any surviving field would be indifferent to all of them. `only_name`
tests that directly: the 2–4 word name and nothing else. It is the worst variant the matcher measured
(**−0.105**) and it is what `embedding_utils.get_use_case_text` falls back to today, so it is the
floor the `scripts/*.py` path actually ships.

| cell | reader d AUC | d F2@own | d fraction-read | collections worse than floor |
|---|---|---|---|---|
| `full` (reference) | 0.000 | 0.000 | 0.000 | 0 of 8 |
| **`only_name`** | **−0.038** | −0.018 | **+0.039** | **5 of 8** |
| `shuffled` (wrong topic) | −0.282 | −0.297 | −0.158 | 7 of 8 |

**It lands between the two anchors, and that is the informative outcome.** −0.038 clears the 0.03
floor and is worse than the floor on 5 of 8 collections, so **the brief's prose is genuinely worth
something to the reader — about 0.04**. But it is nowhere near the −0.282 of a wrong brief.

*ELI18: think of the brief as an instruction sheet with several sections that all describe the same
topic in different words. Tear off everything except the title and the model gets slightly worse
(0.04). Scribble over any one section and it does not get worse at all — the other sections still say
what the topic is. Swap the whole sheet for a different project's and it falls apart (0.28).*

**So the mechanism behind §2's null is redundancy.** The four degradations each corrupt one part of
the brief — `vague_objective` only `objective`, `fluff_replace` only the two prose fields,
`conflicting` and `keyword_flood` only the term lists — and in every case the remaining fields still
carry the topic. **Which means §2 should not be read as "spec quality does not matter to a reader".**
It should be read as: *no single field is load-bearing for a reader, because the same information is
in several of them.* A degradation that corrupted **all** fields at once has not been measured, and
`only_name` is the closest thing to it here.

⚠️ Note `only_name` **increases** fraction-read by 3.9 points while lowering F2@own by 0.018 — given
only a topic label the reader flags *more* papers, less precisely. That is the kind of failure only a
reader can exhibit and a ranker cannot, and it is the one argument left for a reader-side critic.

🟡 **One collection is consistently helped by every degradation**, which is worth recording:
`muthu_2021` gains +0.112 (`conflicting`), +0.117 (`fluff_replace`) and +0.125 (`vague_objective`).
Read as a judgement call: its shipped brief appears to *mislead* this reader, and damaging it removes
the misleading part. n=1, exploratory, and it is the kind of thing the foreign-brief probe was built
to detect.

---

## 3. The three pre-registered bars

**H1 — FAIL.** Required `conflicting` to cost the reader **≥0.03 more AUC than it cost the matcher**,
on ≥5 of 8 collections. It cost the reader **+0.016** — i.e. the reader did marginally *better* with a
self-contradicting brief — against the matcher's −0.014. It clears the gap on **1 of 8**.

This was the plan's headline hypothesis and the strongest prediction in it, drawn from S-AL's measured
0.828 → 0.772 when a hand-written policy contradicted its own labels. It does not reproduce here.
🟡 **Judgement call on why, and it matters for what to build:** S-AL contradicted the brief *in prose*
— a policy sentence naming as positive the category its labels rejected. This variant contradicts it
*in the term lists*, by moving `terms_exclude` into `terms_must_include`. If the reader largely ignores
the term-list fields, the two manipulations are not the same experiment, and that reading is
independently supported: `keyword_flood` also does nothing (−0.007), and `CONTEXT.md` §6 already
rejects "LLM-written `terms_*` for an LLM reader" on separate evidence. **S-AL's finding is not
refuted by this; it is un-addressed by it.** Testing it properly needs a prose-contradiction variant,
which this ablation does not contain.

**H2 — FAIL, both halves.** Required `fluff_replace` and `vague_objective` to cost the reader **≥0.03
AUC**. They cost **+0.003** and **+0.010**. Both are worse than the floor on 2 of 8 and better than it
on 1 and 2 respectively.

The contrast with the matcher is the interesting part: at zero labels those same two variants cost the
matcher **−0.177** and **−0.209** on the feature they damage — the cold-start prose features are driven
*below chance*. So the plan's framing was upside down. It predicted prose damage would hurt the reader
at *every* label count, on the theory that "a reader consumes instructions, and there are none left".
In fact **prose damage is a catastrophe for the matcher at cold start and free for the reader**.

**H3 — PASS.** Required `keyword_flood` to cost the reader **less** than it cost the matcher, on ≥5 of
8. Reader **−0.007** against matcher **−0.072** fitted (and −0.106 unfitted worst). The reader is
essentially unmoved by padding a term list with `used`, `using`, `analysis`, `data`; the matcher loses
0.072 and is worse than the floor on **6 of the 7 set-A collections where the fitted arm is
computable** (`sep_2021` has only 24 train positives, so its score is undefined, not zero).

**Stop rule: fired, and honoured.** The plan pre-registered: *"Stop and report if stage 1 shows all
four variants inside 0.03 of each other — that is itself the answer (a reader is insensitive to spec
form) and buying stage 2 adds nothing."* The spread is 0.023. **Stage 2 was not bought**, which saves
$3.30 and — more valuable — leaves `benchset_v1_large_set_b` unspent. Overriding a pre-registered stop
rule because the null is interesting would be the exact behaviour the rule exists to prevent.

---

## 4. 🟢 What this changes

**D43's spec linter needs one critic, not two — and it is the matcher.** The plan's premise was that
the two consumers fail in mirror-image ways and neither notices the other's failure, so a linter needs
both. Half of that is confirmed and half is inverted:

| failure mode | matcher notices | reader notices |
|---|---|---|
| generic-keyword padding | **yes** (−0.072 set A, −0.031 set B) | no (−0.007) |
| too few must-include terms | **yes** on set A (−0.043), not on set B (−0.010) | untested |
| fluff / vague prose | **yes, catastrophically, at 0 labels** (−0.177 / −0.209) | no (+0.003 / +0.010) |
| self-contradicting term lists | barely (−0.014 / −0.009) | no (+0.016) |
| **nothing but the topic name** | **yes** (−0.105) | **yes, modestly** (−0.038, 5 of 8) |
| wrong topic entirely | **yes** | **yes, decisively** (−0.282) |

**Every row the reader notices, the matcher notices too — and notices harder.** There is no failure
mode in this ablation visible *only* to the reader on ranking quality, so for **ranking** a
two-critic linter buys nothing over a one-critic linter. That is worth the $1.32 to know, and it
simplifies the guidance: **lint the spec against the matcher.**

**But the reader keeps one job, and `only_name` is what shows it.** Stripped to a topic label the
reader's *ranking* only slips 0.038 while its **fraction-read rises 3.9 points and F2@own falls
0.018** — it flags more papers, less precisely. A matcher has no operating point at all, so no
matcher-side check can produce that number. So:

- a check that says *"this brief is badly written"* → **matcher**, and the reader adds nothing;
- a check that says *"this brief will make the model over-flag"* → **reader**, and only the reader.

That is a narrower second critic than D43 assumed, aimed at the operating point rather than at spec
quality. It is not the two-critic linter the plan set out to justify, and it is not nothing.

**For the sibling repo:** `DATA_BRIEF.md` §4.10's finding 5 — *"contradictions are invisible to
matchers and expensive to readers … which is why a spec linter needs both critics"* — should be
narrowed. Contradictions **in term lists** are invisible to both. S-AL's prose-contradiction result
stands on its own evidence and is not tested here; the sentence should say so rather than resting on
`conflicting`.

---

## 5. Cost, and the trap that did not cost anything

| stage | cells | rows | fresh spend |
|---|---|---|---|
| 1 — set A subsample, 4 decisive variants | `full`, `conflicting`, `fluff_replace`, `keyword_flood` | 1,998 each | **$0.88** |
| 1b — H2's untested half + the floor | `vague_objective`, `shuffled` | 1,998 each | **$0.24** |
| 1c — the mechanism diagnostic | `only_name` | 1,998 | **$0.20** |
| 2 — set B held-out | *not bought — stop rule* | — | **$0.00** |
| | | | **$1.32** of $10 |

**`only_name` was bought after the stop rule fired, and that is not a violation of it.** The rule
governs whether to buy *stage 2* — a confirmatory run on the clean surface — once the four variants
land inside 0.03 of each other. `only_name` is a $0.20 diagnostic on the *already-burned* surface,
asked to explain why the null happened rather than to test H1–H3 again. It changed the conclusion
from "a reader is insensitive to spec form" to "no single field is load-bearing because the signal is
redundant", which is a materially different thing to tell someone building a linter.

**`vague_objective` was not in the plan's four-variant list, and H2 names it.** Four cells would have
left a pre-registered hypothesis half-tested, and the two are not interchangeable: `fluff_replace`
swaps real prose for confident-sounding generic prose, `vague_objective` swaps it for an honest
one-liner that names nothing selectable. $0.28 for a testable hypothesis instead of an untestable one.

⚪ **The cache trap, and it was live.** gemma's set-A `P2` own-brief cell was already paid for under
tag `P2|P2-v1|brief-v1`, and `score_frame` appends a brief tag *only* when `brief_map is not None`.
Routing `full` through `brief_map=` with any tag would have changed the key and re-bought ~$1.32 of
identical answers — which `score_frame`'s own docstring records having happened before, on a 12-cell
grid. `full` is therefore scored with `brief_map=None` and verified at **1,998/1,998 cached, $0.00**.
The run refuses to proceed if that check fails, because a moved key means nothing is comparable to the
published set-A numbers *and* every later stage costs double.

⚪ **A second one, caught by the same check.** The derangement was initially re-rolled locally rather
than imported, which scored **36% cached instead of 100%** — a derangement is one of many
permutations, so a different RNG gives different pairings and different prompts. Worse than the cost:
it would not have been the same floor the published 0.498 refers to. `run_llm_screening.
shuffled_brief_map` is imported instead.

---

## 6. Caveats — read before quoting any number above

1. **One model family, unreplicated.** `gemma-4-31b-it` only. It is the *weakest* of the four models
   on set A, which makes a null here conservative rather than flattering, but a stronger reader might
   be more sensitive to spec form, not less.
2. **Set A only, and set A is burned.** Five selection passes. This is the exploratory surface by the
   plan's own accounting. The null is the finding *on set A*; it is not confirmed on clean data, and
   the stop rule is why. A reader-sensitivity result would have been worth confirming; a null on the
   surface that has been tuned hardest toward showing effects is the harder place to find a null,
   which is why stopping is defensible rather than merely cheap.
3. **Tie fraction 0.94–1.00.** Verbalised scores are lumpy and AUC is coarse on them. Small
   differences are genuinely hard to resolve here — mitigated, not eliminated, by the 0.282 floor.
4. **The subsample is small per collection.** ~250 rows and 271 positives across 8 collections;
   `sep_2021` has 54 rows and `nelson_2002` 72. Per-collection AUC is noisy, which is why win counts
   are reported beside every mean and why stage 1 was pre-registered as estimates-only.
5. **`conflicting` is a term-list manipulation, not a prose one.** See H1. It does not test S-AL's
   result and must not be quoted as refuting it.
6. **Five of fourteen variants bought** (`full`, `conflicting`, `fluff_replace`, `vague_objective`,
   `keyword_flood`) plus `only_name` as a diagnostic and `shuffled` as the floor. The other nine are
   unbought and out of budget (~$6.6 more on set B).
7. **The matcher comparators come from two different label counts.** Neither is label-matched to a
   zero-shot reader; both are printed for that reason.
