# Query-conditioned features, embedding choice, and the recall lens

**Audience:** anyone picking up TIRI's feature-engineering or modelling work. Assumes
`README.md` and `CONTEXT.md`. Every number below is reproducible from the three scripts
listed in §8.

**Scope:** why leave-one-use-case-out collapses and what that actually means; a new
feature block that reads the use-case brief, and the control that tests it; the three
candidate embedding models re-scored on recall rather than ranking area; and an external
check on SYNERGY that **partly refutes** the headline claim.

---

## 1. ELI5 — what is this about?

> We have ~2,900 papers across 6 research questions, each marked relevant or not by an
> analyst. We want a model that ranks papers so a human reads the good ones first.
>
> The models were doing fine on questions they'd seen and falling apart on new ones. The
> assumption was "too many features, so it's memorising." That turned out to be the wrong
> diagnosis, and the right one changes what to build.
>
> **The model wasn't memorising noise. It was learning which research question a paper
> belonged to.** Give it a paper's numbers and it can tell you "that's a cement paper"
> 96 times out of 100. So what it really learned was "cement papers are relevant" — true
> for the cement question, useless for a new one.
>
> The fix that follows: stop asking "is this paper good?" and start asking "is this paper
> good **for this brief**?" The briefs were sitting in the data, unread.
>
> That worked on our own data, clearly. It did **not** reproduce on an external dataset —
> §6 explains why, and it's a finding about how briefs are written, not a wash.

---

## 2. TL;DR

**① The transfer collapse is structural, not statistical.** A classifier reads
`use_case_key` off the paper embedding at **96.2%** accuracy. Cross-use-case training
therefore learns use-case identity. Ten brief-relative scalars beat a 384-dimension
embedding on leave-one-use-case-out (**0.642 vs 0.537**). It was never the feature count.

**② At zero labels, a cross-domain model is worse than no model.** Unsupervised
cosine-to-brief scores **0.082 WSS@95 / 0.693 ROC-AUC** on a held-out use case; the best
supervised cross-domain model manages **0.042 / 0.610**. From ~25 in-silo labels the
supervised model overtakes it and keeps climbing.

**③ The three embedding models are indistinguishable on our data, and not on SYNERGY.**
Within-silo, seed noise (sd 0.015–0.027 WSS) exceeds the entire spread between models
(0.027). At SYNERGY's realistic prevalence they separate, and **Qwen3-8B wins**.

**④ PCA-64 hurts within a silo** (−0.008 to −0.014 WSS, all three models), reversing the
compress-then-concatenate finding in `wf_featureengineering_review.md` §6.2 — that was a
*transfer* result and does not carry to the production surface.

**⑤ The lexical block does not replicate on SYNERGY, and its control fails there.** The
most important caveat in this document; §6.

> **ELI5 on the metrics.** **ROC-AUC**: 0.5 = coin flip, 1.0 = perfect ranking.
> **WSS@95**: the share of the pile a reviewer can skip while still finding 95% of the
> good papers. Its ceiling is `0.95 × (1 − prevalence)` — in a pool that's 77% relevant
> there's almost nothing to skip, so small numbers there aren't failure.

---

## 3. The diagnosis

`reports/wf_featureengineering_review.md` framed the holdout collapse as overfitting from
feature count. Three measurements say otherwise.

**The embedding is a use-case fingerprint.** 6-way classification of `use_case_key` from
the 384-d paper vector: **96.2%** accuracy against a 19% majority baseline. Features that
identify the use case this precisely cannot help on a use case the model has never seen —
relevance is defined *relative to a brief*, and a paper-only feature has no access to one.

**Removing the offset doesn't fix it.** Per-use-case mean-centring the embedding: LOGO
0.532 → **0.529**. Centring removes a constant; the discriminative direction is different
in every pool, so there is nothing to recover. You cannot subtract your way out.

**Fewer features aren't the answer; different features are.** Ten brief-relative scalars
(TF-IDF cosine to five brief fields plus within-pool ranks, fitted inside the fold) reach
**0.642** LOGO ROC-AUC against the 384-d embedding's **0.537**. Ten beat 384.

This also explains three earlier observations under one mechanism: PCA helping transfer
(it destroys use-case-identifying directions), metadata raising dev score while lowering
holdout (provenance *is* use-case identity), and term overlap being the only one of eleven
proposed features that worked (the only one that read the brief).

---

## 4. Tier 1b — the lexical block, and its falsification control

`scripts/lexical_features.py` emits 22 columns per paper, all brief-relative: Okapi BM25
of the paper against five brief fields (`objective`, `problem_statement`,
`terms_must_include`, `terms_nice_to_have`, domain), within-pool percentile ranks of each,
term-overlap counts and fractions, and length-normalised variants.

**Nothing is fitted on labels**, so the block is safe to materialise. BM25 IDF and the
percentile ranks are computed *per use-case pool* — transductive but production-realistic,
since a silo's pool is known at scoring time.

### The control

Features like these can score well for a boring reason: if they secretly measure abstract
length, they'd work against *any* brief. So `build_lexical_features` takes a `brief_map`
and can be rebuilt against deliberately wrong briefs.

| LOGO mean | ROC-AUC | PR-AUC |
|---|---|---|
| paper embedding (384-d) | 0.537 | 0.598 |
| **Tier 1b, real briefs** | **0.642** | **0.696** |
| Tier 1b, **shuffled briefs** | 0.488 | 0.577 |

Real beats wrong on **5/6** use cases, mean gap **+0.155**, between-derangement sd 0.037.
The wrong-brief version lands *below chance* — there is no generic residual. On our data
this is as clean a pass as the test can give.

### Two side-questions closed

- **BM25 alone is essentially the whole block**: 0.637 of the 0.642. Ship ~10 columns, not 22.
- **Term overlap is not an abstract-length proxy** — the open question from
  `wf_featureengineering_review.md` §8. Length features alone reach 0.550; removing them
  costs 0.004.

---

## 5. Embedding models under the recall lens

All three re-scored on WSS@95, on both fold surfaces. `scripts/run_embedding_recall_comparison.py`.

### Leave-one-use-case-out — chooses defaults, is not a production number

| variant | mean WSS@95 | mean ROC-AUC |
|---|---|---|
| **jasper: cosine-to-brief (0 labels)** | **0.082** | **0.693** |
| qwen3-4b: cosine-to-brief | 0.080 | 0.687 |
| qwen3-8b: cosine-to-brief | 0.077 | 0.662 |
| lexical only (Tier 1b) | 0.051 | 0.642 |
| *best supervised cross-domain* | 0.042 | 0.610 |
| *worst supervised* | 0.016 | 0.550 |

Zero-label approaches win on **4/6** use cases. Training across silos is actively harmful.

### Within-silo — the production surface

| variant | mean WSS@95 | mean ROC-AUC |
|---|---|---|
| Jasper + Qwen3-4B concat (the bake-off's pick) | 0.166 | 0.842 |
| qwen3-8b: PCA-64 + lexical | 0.159 | 0.836 |
| qwen3-4b: embedding | 0.153 | 0.835 |
| qwen3-8b: embedding | 0.151 | 0.835 |
| jasper: embedding | 0.147 | 0.820 |

**Seed-to-seed sd is 0.015–0.027 and the whole spread is 0.027.** Nothing here separates.
The bake-off's recommendation survives — nominally top on two of three metrics — but as a
tie, not a demonstrated win. Worth noting the mechanism differs from why it was chosen:
Jasper is *last* of the three alone, yet adds value in the concatenation. That's diversity,
not quality.

### Warm-start — WSS@95 by in-silo label budget

| labels | qwen3-4b embedding | cosine-to-brief floor |
|---|---|---|
| 0 | — | 0.080 |
| 25 | **0.100** | 0.080 |
| 50 | 0.133 | 0.080 |
| 100 | 0.149 | 0.080 |
| 200 | 0.159 | 0.080 |

The crossover is at or below 25 labels. **The operating rule: cosine-to-brief until ~25
in-silo labels exist, supervised in-silo after, never cross-customer.**

### PCA-64 hurts within a silo

Raw beats PCA-64 on all three models (jasper −0.014, qwen3-4b −0.009, qwen3-8b −0.008).
`wf_featureengineering_review.md` §6.2 found the opposite, and both are right: compression
helps *transfer* because it destroys use-case-identifying directions, and hurts *within a
silo* because there is nothing to protect against and information is simply lost. Do not
carry PCA into production on the strength of a LOGO result.

---

## 6. SYNERGY — where the headline claim does not replicate

Three published systematic reviews, 1.7–14.8% included, labelled by researchers with no
connection to this project. The only prevalence-realistic surface available.
`scripts/run_synergy_recall_validation.py`.

### The embeddings look far better here

| variant | mean WSS@95 | mean ROC-AUC |
|---|---|---|
| qwen3-8b: embedding | **0.517** | **0.888** |
| jasper: embedding + lexical | 0.510 | 0.876 |
| qwen3-8b: embedding + lexical | 0.458 | 0.883 |
| qwen3-4b: embedding + lexical | 0.451 | 0.879 |
| jasper: embedding | 0.438 | 0.866 |
| qwen3-4b: embedding | 0.375 | 0.874 |
| lexical only (Tier 1b) | 0.052 | 0.613 |

WSS@95 of 0.38–0.52, against 0.13–0.17 on our own pools — because at 1.7–15% prevalence
the ceiling is 0.81–0.93 rather than 0.22–0.70. **Recall metrics only discriminate when
there is something to skip.** Our enriched pools are a poor surface for judging a
recall-oriented system, and this is the strongest argument for keeping SYNERGY in the loop.

**Qwen3-8B wins here**, and Qwen3-4B is *last* of the three — the reverse of its in-repo
standing. So the model ordering is not stable across prevalence regimes, which is a
further reason not to treat the 0.004-margin bake-off decision as settled.

### The control fails

| review | real brief | wrong brief | gap |
|---|---|---|---|
| Sep_2021 | 0.552 | 0.570 | −0.019 |
| Menon_2022 | 0.535 | 0.566 | −0.031 |
| van_der_Waal_2022 | 0.753 | 0.739 | +0.015 |

Real briefs beat wrong ones on **1/3** reviews, mean gap **−0.012**. On SYNERGY the block
is not reading the brief, and it performs near chance (0.613 ROC-AUC) regardless.

**The most likely explanation, and it is testable.** SYNERGY carries no curated term lists,
so the block loses BM25-on-`must`/`nice` and every overlap column — the components the
ablation showed carry the signal. What remains is BM25 against the review's title and
abstract, and those are a *report of what a review did*, not a statement of what to
include. TIRI's briefs are written to select papers; a published review's abstract is
written to summarise findings. Different genres, and only one of them discriminates.

**What this does and does not license.** It does not overturn §4: that control passed 5/6
with five derangements on briefs written as selection criteria. It does establish that the
query-conditioned advantage is **contingent on brief format**, not automatic — and that a
brief lacking explicit inclusion terminology may buy nothing at all. That is directly
actionable for the brief-format experiment running elsewhere: *`terms_must_include` and an
objective phrased as selection criteria are load-bearing, and `constraints_scale` was never
filled in once across six use cases.*

Honest limits: three reviews, one rotation (with three items there are only two
derangements), and a block running at reduced strength. This is a warning flag, not a
refutation, and it should be re-run when more SYNERGY reviews are pulled in.

### Random label bootstraps are unusable at low prevalence

Share of random *n*-label draws containing no positive at all, so nothing can be fit:

| review | n=25 | n=50 | n=100 | n=200 |
|---|---|---|---|---|
| Sep_2021 (14.8%) | 0.05 | 0.00 | 0.00 | 0.00 |
| Menon_2022 (7.6%) | 0.15 | 0.00 | 0.00 | 0.00 |
| **van_der_Waal_2022 (1.7%)** | **0.57** | **0.28** | **0.22** | 0.05 |

At 1.7% prevalence a 25-label draw contains 0.4 positives on average and **more than half
of such draws are unusable**. The "~50 informative labels" bootstrap budget in
`wf_ensemble_report.md` §4 is only viable if those labels are **actively selected** — for
example by labelling the top of a cosine-to-brief ranking — not randomly sampled. This is
the strongest practical argument for shipping the zero-label ranker: not because it scores
well, but because it is what makes the first labels findable at all.

---

## 7. What to do

1. **Ship two modes.** Cosine-to-brief below ~25 in-silo labels; supervised in-silo above.
   Never a model trained on other customers.
2. **Use the zero-label ranker to source the bootstrap labels**, not random sampling. §6.
3. **Keep Jasper + Qwen3-4B concat if inertia is cheap, but record it as a tie.** If
   choosing fresh on the prevalence-realistic evidence, **Qwen3-8B alone** is the better bet.
4. **Drop PCA from the production path.** Keep it only for transfer-shaped questions.
5. **Keep Tier 1b, but as a small additive, not a lever** — and re-validate it against any
   new brief format before assuming it transfers.
6. **Feed §6 back into the brief-format experiment.** Curated inclusion terminology is what
   makes the block work; descriptive prose isn't.
7. **Ring-fence the 23 unused SYNERGY reviews.** LOGO has been selected against many times
   and is no longer unbiased; that is the only clean surface left.

---

## 8. Limits — what this does *not* establish

- **Dense cosine-to-brief was not tested on SYNERGY.** Only paper vectors are cached;
  three brief vectors would need a Modal GPU spin-up. The lexical BM25 analogue was used
  instead, and it is weaker.
- **Three SYNERGY reviews, one brief rotation.** §6's control result rests on a very small
  sample and a degraded feature block.
- **No author grouping on SYNERGY** (the export carries no author column), so those
  within-review numbers are slightly optimistic relative to the in-repo ones.
- **Per-use-case percentile ranks are transductive.** Label-free and realistic, but they
  shift if production pool composition differs from the dev pools.
- **All supervised numbers use plain LogisticRegression.** Deliberate — a difference is
  then about features, not about which arm got a better learner — but no tree/boosted model
  was tried on these feature blocks.
- **`solar_leo` remains a corpus defect**, not a modelling failure. Its labels track
  publication year because the pool was seeded from citation-ranked canon.

---

## 9. Where the evidence lives

| What | Where |
|---|---|
| Lexical block + shuffled-brief control + warm-start | `scripts/lexical_features.py`, `scripts/run_tier1b_control.py` → `reports/wf_tier1b_lexical_control.md` |
| Three models on WSS@95, both fold surfaces | `scripts/run_embedding_recall_comparison.py` → `reports/wf_embedding_recall_comparison.md` + 3 CSVs |
| SYNERGY at realistic prevalence | `scripts/run_synergy_recall_validation.py` → `reports/wf_synergy_recall_validation.md` |
| Shared fold/metric helpers (reuse, don't re-implement) | `scripts/fold_pipeline_utils.py`, `scripts/embedding_utils.py` |
| Project context, negative-results register, noise floors | `CONTEXT.md` |
| The reviews this supersedes in part | `reports/wf_featureengineering_review.md` §6.2, `reports/wf_embedding_bakeoff.md` §8 |
