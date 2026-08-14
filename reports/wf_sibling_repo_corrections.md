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

**Action:** §4.11's `objective` MVP row can be stated more strongly, not less. *"Fluff/vagueness →
below chance"* holds on both surfaces, and the cost of writing no objective at all is **−0.263** on
clean data rather than −0.151.

*ELI18: the number that matters before anyone has labelled anything is "how well does the brief's
own prose sort papers on its own". On fresh data, deleting the objective hurts that **more** than we
first measured — it is the single most load-bearing thing in the whole brief at the moment a project
starts.*

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
- **But this does not refute S-AL.** S-AL's 0.828 → 0.772 contradicted the brief **in prose** — a
  policy sentence naming as positive the category its own labels rejected. TIRI's variant contradicts
  it **in the term lists**. If a reader largely ignores term-list fields, these are not the same
  experiment — and two separate results say it does: `keyword_flood` costs a reader −0.007, and
  `CONTEXT.md` §6 already rejects "LLM-written `terms_*` for an LLM reader" on independent evidence.

**Action:** rest finding 5 on S-AL's own prose evidence and **say that explicitly**, rather than on
the `conflicting` variant, which does not support it. The stronger overall claim from the reader arm:

| failure mode | matcher notices | reader notices |
|---|---|---|
| generic-keyword padding | **yes** (−0.072 / −0.031) | no (−0.007) |
| fluff / vague prose | **yes, at 0 labels** (−0.177 / −0.209) | no (+0.003 / +0.010) |
| self-contradicting term lists | barely (−0.014 / −0.009) | no (+0.016) |
| nothing but the topic name | **yes** (−0.105) | yes, modestly (−0.038) |
| wrong topic entirely | **yes** | **yes, decisively** (−0.282) |

**Every failure mode the reader notices, the matcher notices harder.** So for **ranking quality**,
D43's linter needs one critic and it is the matcher. **The reader keeps exactly one job**: stripped to
a bare topic name it *over-flags* — fraction-read **+3.9pp** while F2@own falls 0.018. A matcher has no
operating point, so no matcher-side check can produce that number. That is a narrower second critic
than D43 assumes, pointed at the operating point rather than at spec quality.

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

## 6. ⚪ Two things that do not need a correction, but change what is buildable

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

All bars were pre-registered in `wf_spec_quality_plan.md` before any measurement, and that file
carries a dated Amendment recording five premises corrected **before** running — plus a closing
scoreboard against its own predictions, which it got 2 of 4 right. Total API spend for everything
above: **$1.32**.
