# Corrections for `academic_agent` — five claims this week's measurements changed

Status: **hand-off, 2026-08-14.** For the owner of `academic_agent` to apply; nothing in that repo
has been edited from here (it is read-only from TIRI by convention).

**Why this exists.** `DATA_BRIEF.md` §4.11 is about to become a **form that analysts fill in**, and
parts of it rest on set-A numbers that this week's clean-surface runs changed. One `NUMBERS.md` row is
registered `LIVE+SQL` — i.e. shippable today — on a finding that does not survive. Correcting a brief
before it becomes an interface is cheaper than correcting an interface.

**The good news first, because it is the larger half.** The *cold-start* half of §4.10 does not merely
replicate on clean data — **it gets stronger**. It is the *after-labels* half that halves. So §4.11's
headline recommendation (`objective` must read like a paper) is better supported than claimed, and its
term-list recommendations are weaker.

**Evidence base.** `benchset_v1_large_set_b` — 7 collections, 82,350 rows, 1.87% positive, **never
scored by anything before this week**. Nothing in the repo had opened the file; `grep set_b
scripts/*.py` returned zero hits. Every number below is a paired A-vs-B comparison over the identical
14 brief variants and the identical instrument.

⚠️ **The 0.03 rule applies throughout.** `CONTEXT.md` §5: seed-to-seed noise on within-silo ROC-AUC is
~0.010, and gaps under ~0.03 are **not established**. Several numbers below cross that line in one
direction and not the other, and that is usually the whole point.

---

## 1. 🟢 The cold-start findings replicate and strengthen — strengthen §4.11's MVP row

The unfitted, zero-label arm — one feature, no labels, nothing able to compensate — is the one that
reads the brief's prose directly. Damage to it, `d` on the worst-affected feature:

| variant | set A | set B | verdict |
|---|---|---|---|
| `no_obj` — no objective | −0.151 | **−0.263** | replicates, **worse** on clean data |
| `only_name` — the 2–4 word name alone | −0.151 | **−0.263** | replicates, **worse** |
| `vague_objective` | −0.209 | −0.194 | replicates |
| `fluff_replace` | −0.177 | −0.169 | replicates |
| `no_nice` — no nice-to-have terms | −0.105 | **−0.215** | replicates, **twice as bad** |
| `must_only_one` — one must-include term | −0.103 | **−0.175** | replicates, worse |

🔴 **But one specific claim in §4.10 finding 1 does NOT replicate, and it is the most quotable one.**
*"Fluff and vagueness drive the objective feature below chance (0.442–0.473 against 0.651), which is
worse than deleting the field"* is **set-A-specific**. Read the same feature in absolute terms:

| `objective` feature, absolute (0.500 = coin flip) | set A | set B |
|---|---|---|
| real prose (`full`) | 0.651 | **0.763** |
| fluff **appended** to real prose | 0.633 | 0.758 |
| generic fluff **replacing** it | **0.473** 🔴 below chance | **0.594** |
| vague one-liner | **0.441** 🔴 below chance | **0.569** |
| field deleted / name only | 0.500 | 0.500 |

On clean data neither variant goes below chance, and **both are better than deleting the field**, not
worse. The ordering on set B is monotone and unsurprising — real prose > diluted prose > generic prose
> vague prose > nothing — where on set A the bottom two dipped *below* nothing.

⚠️ **And a measurement caveat that matters for how §1's deltas are read.** `no_obj` and `only_name`
both land on **exactly 0.500** because deleting the field makes that feature uncomputable, so it
degenerates to a coin flip. Their "delta" is therefore just `full − 0.500`, which mechanically grows
with how good `full` was. So **−0.263 on set B is not "deletion hurts more on B"** — it is *"the
objective feature is worth 0.263 above chance on B"*. Same for `no_nice` (−0.215) and the cold-start
half of `must_only_one`. Read those rows as **what the field is worth**, not as damage.

**Actions:**

- §4.11's `objective` MVP row **stands and can cite two surfaces**: the field is worth **+0.151 (A) /
  +0.263 (B)** above chance at zero labels. It is the single most load-bearing field in the schema on
  both surfaces, by a wide margin.
- §4.10 finding 1's *"below chance … worse than deleting the field"* → **retract, or scope to set A.**
  On clean data any prose beats no prose. The practical guidance barely changes ("write a real
  objective") but the *rhetorical* claim does, and it is the one a designer would quote.

*ELI18: the objective — the plain-English description of what you're looking for — is the single most
valuable thing in the brief before anyone has labelled anything, worth about 0.26 on a scale where 0
is a coin flip. Writing it badly costs you most of that. On our older dataset writing waffle actually
scored **worse than leaving the box empty**, which was a great line; on fresh data it doesn't, and
waffle is merely bad rather than actively misleading.*

---

## 2. 🔴 The after-labels term findings roughly halve — revise §4.10 findings 2 and 3, and §4.11

The fitted arm at 60 labels. `d`, with the count of collections falling more than 0.03 below `full`:

| variant | set A | set B | verdict |
|---|---|---|---|
| `keyword_flood` — must/nice padded with generics | **−0.072** (6 of 7) | **−0.031** (3 of 7) | halves; now at the noise floor |
| `no_must` — no must-include terms | **−0.043** (4 of 7) | **−0.010** (1 of 7) | **does not replicate** |
| `only_name` | −0.105 (4 of 7) | −0.045 (3 of 7) | halves; still the worst variant on both |
| `must_only_one` | −0.012 (3 of 7) | −0.010 (1 of 7) | inside the floor on both |
| `conflicting` | −0.014 (1 of 7) | −0.009 (1 of 7) | replicates: cheap for a matcher either way |

**Why set A overstated it: one collection.** `synergy_leenaars_2020` alone contributes
`keyword_flood` **−0.220** and `no_must` **−0.149**. Remove it and set A's means collapse toward set
B's. Set B's smaller `keyword_flood` mean is likewise carried by `synergy_walker_2018` (−0.111),
which is 56% of the set. **On both surfaces this is a one-collection effect that the win count did
not expose**, because 0.03 is a low bar relative to the spread.

**Actions:**

- **§4.10 finding 3** — *"`terms_must_include` is the single load-bearing field for the fitted
  matcher (−0.043, 4 of 7)"* → **retract**. On clean data it is **−0.010 on 1 of 7**, inside the
  noise floor. The field may still be worth requiring, but not on this evidence.
- **§4.10 finding 2** — *"Too many bad keywords is worse than no keywords"* → **keep the ordering,
  drop the magnitude**. Padding (−0.031) is still worse than an empty field (−0.010) on set B, so the
  direction survives twice. But both now sit at or inside the noise floor, so "measurably worse" is
  too strong for the clean surface.
- **§4.11 MVP row `terms_must_include`** — the evidence column reads *"The one load-bearing field
  after labels (−0.043, 4 of 7)"*. Replace with the paired figure and its caveat. Do **not** drop the
  field: §1 shows `must_only_one` costs **−0.175** at cold start on B, so a term list of one term is
  genuinely bad — it is the *after-labels* claim that fails, not the field.
- **§4.11 Never row `generic keywords`** — *"−0.072, worse than an empty field"* → cite both surfaces.
- **§4.11 Never row `a 2–4 word topic name alone`** — *"−0.105"* → **−0.105 / −0.045**. It remains the
  worst variant measured on both, so this row stands.

---

## 3. 🔴 Remove `NUMBERS.md` N33 from the screen — the foreign-brief alarm does not survive

N33 currently ships at `LIVE+SQL`: *"use case Y's brief ranks your corpus better than your own"*, with
the `soil_microbiome` 0.603-own / 0.792-foreign pair as evidence. §2.6 lists it as one of three
brief-quality instruments. Both were measured this week and neither holds.

**It is not a brief property.** Rebuild the same 34×34 brief × corpus matrix in **BM25 space** instead
of embedding space and the flagged set does not carry over: 3 of 5 survive, **8 new collections flag
that the embedding called perfectly healthy** (against a pre-registered bar of ≤2), and the two
margins correlate at only **rho 0.33**. `soil_microbiome`'s headline **−0.190 becomes −0.021** —
inside the noise floor, i.e. no detectable problem. A defect in written text should be visible to any
reader of that text; this one is visible to one embedding.

**And it is not a labelling-cost forecast either**, which was the fallback caption TIRI briefly
proposed. On set A the margin predicted labelling gain at rho −0.929, which looked strong enough to
ship under a different sentence. It does not replicate:

| surface | margin from | rho | 95% CI | rho given prevalence |
|---|---|---|---|---|
| A (burned) | jasper | −0.929 | −1.000 to −0.412 | −0.929 |
| A (burned) | **qwen4b** | **−0.571** | **−1.000 to +0.333** | −0.429 |
| A (burned) | lexical | −0.286 | −0.887 to +0.686 | −0.286 |
| **B (clean)** | **jasper** | **−0.464** | **−1.000 to +0.765** | **+0.143** |
| B (clean) | qwen4b | −0.036 | −0.887 to +1.000 | +0.143 |

On clean data nothing clears the bar, and **the sign flips positive once prevalence is held
constant** — prevalence being the confound already measured at rho −0.72 against LOGO transfer. It was
also never embedding-independent on set A: swap jasper for qwen4b on the *same rows* and the
correlation drops to −0.571 with an interval crossing zero.

**Action: delete N33, and remove the foreign-brief row from §2.6's three-instrument table.** §2.6's
other two instruments are unaffected and both still hold — the **shuffled-brief control** passes
decisively (AUC 0.777 → 0.498 lexically, and **−0.282 AUC / −0.297 F2@own** for an LLM reader), and
the **criteria-populated counter** (N23) is untouched. So §2.6 becomes two cheap instruments, not
three. *"A brief a wrong brief could replace is not a brief"* stands; *"a brief a stranger's brief
outperforms is not a brief"* does not.

⚠️ **A methodological note worth carrying, because it nearly fooled TIRI too.** The set-A result
survived dropping any single collection (−0.886 to −0.943), which read as robustness. **Leave-one-out
stability tests whether one data point carries a correlation; it cannot test whether the whole surface
does.** Only a different surface tests that.

---

## 4. 🔴 Narrow §4.10 finding 5 — the two-critic linter is one critic plus an operating-point check

§4.10 finding 5 concludes: *"Contradictions are invisible to matchers and expensive to readers …
which is why a spec linter needs both critics."* The first half is confirmed; the second is not
tested by the experiment cited for it.

- **Matchers: confirmed on clean data.** `conflicting` costs a matcher −0.014 (set A) and −0.009
  (set B). BM25 cannot notice a term list contradicting itself, for a structural reason.
- **Readers: measured, and it went the other way.** Moving `terms_exclude` into `terms_must_include`
  cost an LLM reader **+0.016** — marginally *better*. It clears the required gap on 1 of 8
  collections.
- **The prose version was then measured directly, and it is a different animal.** ✅ TIRI built
  `conflicting_prose`: a plausible policy sentence appended to the objective asserting that the
  spec's own `terms_exclude` categories are wanted, with that list left intact — so the spec
  contradicts its sibling field *and* its own labels. Result: **reader AUC +0.001** (no ranking
  effect) but **fraction-read +0.120 on 8 of 8 collections** (+1.4 to +34.5pp), recall up on 8 of 8.
  A *fitted* matcher sees **nothing at all** — 0 collections affected on all three surfaces.

🔴 **So finding 5's conclusion is right and its stated reason is wrong.** A spec linter *does* need a
reader-side critic, but not because contradictions cost a reader ranking quality — they do not. It is
because a prose contradiction makes a reader **over-flag**, and that is invisible to every
matcher-side check once labels exist. That is a narrower and more useful claim than the original.

⚠️ **S-AL's number is not reproduced.** S-AL measured 0.828 → 0.772 (−0.056 AUC); this measures
+0.001 AUC plus a 12-point read shift. Read it as *"the effect is real and AUC was the wrong
instrument"* — neither a confirmation nor a refutation. One model family, one corpus, and a
manipulation built from the exclude list rather than a hand-written policy. §4.10 should say so
rather than continuing to cite the AUC figure as though it were replicated.

**Action:** rewrite finding 5 around the operating point rather than around ranking, and cite
`wf_spec_quality_reader.md` §3a for the measurement. The stronger overall claim from the reader arm:

| failure mode | matcher notices | reader notices |
|---|---|---|
| generic-keyword padding | **yes** (−0.072 / −0.031) | no (−0.007) |
| fluff / vague prose | **yes, at 0 labels** (−0.177 / −0.209) | no (+0.003 / +0.010) |
| self-contradicting term lists | barely (−0.014 / −0.009) | no (+0.016) |
| nothing but the topic name | **yes** (−0.105) | yes, modestly (−0.038) |
| wrong topic entirely | **yes** | **yes, decisively** (−0.282) |

**On ranking quality, every failure mode the reader notices the matcher notices harder.** So for
*ranking*, D43's linter needs one critic and it is the matcher.

🟢 **But two rows are genuinely reader-only, and both are about the operating point.** Stripped to a
bare topic name the reader over-flags (+3.9pp read, F2@own −0.018); given prose that contradicts its
own exclusions it over-flags far more (**+12.0pp read on 8 of 8**, and a fitted matcher cannot see it).
A matcher has no operating point at all, so no matcher-side check can produce either number. **That is
the second critic, and it is narrower and better-specified than D43 assumes** — it checks whether the
spec will make the model flag too much, not whether the spec is well written.

🟢 **TIRI shipped this as a linter check**: `exclude_asserted_in_prose` in `scripts/spec_linter.py`, at
**0/34 false positives** on real specs and firing on 8/8 constructed cases. It needs a negation guard —
`synergy_chou_2003`'s objective legitimately contains two of its own exclude terms as "non-cancer pain"
and "non-parenteral" — so port the guard, not just the check.

**Why the reader is so robust, which is the useful mechanism:** stripping the brief to its topic name
costs a reader only −0.038, so the prose is worth ~0.04 in total — but each degradation corrupts *one*
field and the remaining fields still carry the topic. **No single field is load-bearing for a reader
because the signal is redundant across fields.** That is a different claim from "spec quality does not
matter to a reader", and only the first is supported.

---

## 5. 🟡 Add a caveat to §6 — one text-side lever is live, not closed

§6.1 states *"everything tried on the same text has failed, and by similar margins"*, and §6.3
concludes the remaining edge *"is non-text, and the UX is its only source"*. That reasoning is sound
but its premise has an unexamined gap: **every rejected feature added columns *beside* the embedding;
none changed what text goes *into* it.**

Measured this week: feed the embedder **title only** instead of title + abstract, model held fixed.
|difference| **0.027 to 0.056** across four collections, **three of them clearing the 0.03 floor** —
comparable to or larger than most of §6.1's rejected features. And **the sign is inconsistent**: two
collections do *better* without the abstract, two worse. A mean would be +0.007 and would report "no
effect" while hiding four.

**Action:** §6.3's "the edge is non-text" should carry *"…with one text-side lever measured as live
and not yet explored"*. It does **not** change §6.3's advice — the UX is still the best source of a
non-text edge — but it stops the section reading as though the text side is closed.

⚠️ Two honest limits: this used a small CPU model (`BAAI/bge-small-en-v1.5`), **not** the jasper /
qwen4b models behind the shipped `cos_brief_*` columns, so it shows the axis moves and not by how much
the shipped number would move. And the `+venue` arm was dropped as unbuildable —
`papers_benchset_v1.parquet` has no `venue` column.

---

## 6. 🟢 The linter is built, and it is meant to be ported rather than re-derived

`scripts/spec_linter.py` — **11 deterministic, label-free checks**, each carrying the measured cost
that prices it, quoted on all three surfaces. `--self-test` asserts two properties: every check fires
on the exact string whose cost was measured (`FLUFF`/`VAGUE`/`GENERIC` are *imported* from the ablation
script, never restated), and **no check fires spuriously on any of the 34 real specs**.

Four blockers, five warnings, one note, and an **explicit do-not-warn list** — `problem_statement`,
`domain_*` and `terms_exclude` being empty, because they measure 0.000/≤0.009 and a linter that fires
on healthy input teaches the analyst to ignore it (S-FR's own conclusion).

Run over all 34 specs it returns **5 findings, every one on TIRI's own use cases and none on the 28
benchset briefs** — which is the same finding as §1's aside, arriving from the other direction.

**Two things worth porting exactly rather than reimplementing:** the negation guard on
`exclude_asserted_in_prose` (see §4), and the *conjunction* in the instruction-shape detector —
imperative-opening alone fires on 4 of 34 real specs and self-reference alone on 12, while
`(imperative OR self-ref) AND ≥2 evaluative adjectives` fires on the measured bad string and 0 of 34.

## 7. ⚪ Two things that do not need a correction, but change what is buildable

- **`benchset_v1_large_set_b` is now loadable** (`benchset_loader.load_set("b")`, plus
  `drop_ab_crossing` for the 154 papers straddling the A/B boundary). Any other claim in
  `DATA_BRIEF.md` resting on set A can now be checked on a surface that has never been selected
  against, for $0 where no LLM is involved. Given §1 and §2 above — the cold-start half strengthening,
  the after-labels half halving — that is likely worth doing for the §4.6 label-contract numbers too.
- **`nykvist_evcharging` is not a SYNERGY collection**, despite `brief_provenance = review_abstract`.
  Its `objective` is 160 characters against ~1,200 for every SYNERGY collection — someone wrote it.
  Anything averaging over set B's 7 collections should treat it as an exception, and it is one of only
  two non-clinical collections in the whole benchset.

---

## Provenance

| claim | measured by | report |
|---|---|---|
| §1, §2 | `run_spec_quality_ablation.py --set {a,b}` | `wf_spec_quality_ablation.md`, `wf_spec_quality_ablation_set_b.md` |
| §3 specificity | `run_foreign_brief_detector.py` | `wf_foreign_brief_detector.md` |
| §3 validity | `confirm_foreign_brief_validity.py` | `wf_foreign_brief_validity_setb.md` |
| §4 | `run_spec_quality_reader.py`, `analyze_spec_quality_reader.py` | `wf_spec_quality_reader.md` |
| §5 | `run_text_input_precheck.py` | `wf_text_input_precheck.md` |
| §6 | `spec_linter.py --self-test`, `--specs all` | `wf_spec_linter.md` |

All bars were pre-registered in `wf_spec_quality_plan.md` before any measurement, and that file
carries a dated Amendment recording five premises corrected **before** running — plus a closing
scoreboard against its own predictions, which it got 2 of 4 right. Total API spend for everything
above: **$1.32**.
