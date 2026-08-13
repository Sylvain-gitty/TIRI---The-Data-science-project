# LLM screening on benchset set A — findings

**What this is:** the decision doc for the run the pilot said it could not substitute for.
`wf_llm_pilot_findings.md` scoped itself honestly — TIRI's six use cases run 26–77% positive,
the all-positive F2 baseline there is **0.872**, and at that prevalence the metric cannot
separate the ensemble (+0.021) from the best LLM (+0.013). `CONTEXT.md` §4 says the same twice.
`benchset_v1_large_set_a` is the missing surface: **8 SYNERGY collections, 62,229 papers,
1,362 relevant (2.19%)**, where the all-positive floor is **0.101** and F2 discriminates.

**Answer, in one line: a zero-shot LLM does not replace the cold-start ranker, but a brief
distilled from 60 labels ties a supervised model on ranking while reading half as much — and
the brief, not the model, is where the leverage is.**

**Cost:** $16.10 for 89,937 responses across 9 cells (100% parse, 0 HTTP errors, 7
truncations). Rule induction $0.05. The supervised comparison and every baseline cost nothing —
set A's features were already on disk.

**Confidence key:** 🟢 clears the 0.03 noise floor / decisively measured · 🟡 real but under
the floor · ⚪ engineering finding.

---

## 0. How to read any number here

Scoring all 62,229 rows × 4 models is ~$130 and almost all of it buys negatives. Instead every
positive plus a random share of negatives was scored — **9,993 rows per cell** — and reweighted
to population prevalence. Recall and ROC-AUC survive that untouched; precision, F2, WSS@95 and
recall@k need the weight, and dropping it gives a flatteringly wrong answer.

`reports/wf_llm_benchset_a_baselines.md` is the gate, run **before** any spend: the same
metrics computed both ways, three seeds. Max discrepancy **AUC 0.004, WSS@95 0.006, F2 0.017**;
prevalence recovery exact to 4 decimals. Weighted precision/F2/recall@10% independently
reproduce the full corpus to ≤0.026/≤0.042. Every cell scored the identical rows, so
method-vs-method gaps are **paired** and far tighter than the absolute intervals.

Two structural facts that decide how the tables read:

- **`brouwer_2019` is 60% of set A and trivially easy** — 62 positives in 37,401 rows, cosine
  AUC 0.998. It dominates every mean. Win counts, not means (`CONTEXT.md` §5).
- **The bar is the cold-start cosine-to-brief ranker**, not the ensemble. Set A has no shipped
  ensemble, and `wf_llm_pilot_findings.md` §8 already named the cold-start contest the more
  winnable one and the one that would replace something real.

---

## 1. The zero-shot contest: no

All four models, P2, supplied brief, against `cos_brief_qwen4b` (mean AUC 0.763 on all rows):

| Model | mean AUC | beats cosine | loses | inside floor | F2@own | recall | corpus read |
|---|---|---|---|---|---|---|---|
| `nemotron-3-super-120b` | **0.804** | 3/8 | 1 | 4 | 0.389 | 0.552 | 0.121 |
| `qwen3.5-397b` | 0.790 | 3/8 | 1 | 4 | 0.395 | 0.490 | 0.131 |
| `gpt-oss-20b` | 0.783 | 3/8 | 3 | 2 | 0.402 | 0.490 | 0.103 |
| `gemma-4-31b` | 0.774 | 3/8 | 4 | 1 | 0.293 | 0.486 | 0.148 |
| cosine (0 labels) | 0.763 | — | — | — | — | — | — |

**Pre-registered bar: FAIL.** It required beating cosine by >0.03 on ≥6 of 8 collections. Every
model wins exactly 3. The two larger models never lose, which is not nothing, but 3/8 is not
the bar and the bar was fixed before the run.

## 2. 🟢 The pilot's model ranking does not survive the prevalence change

| Model | TIRI AUC (26–77% pos) | rank | set A AUC (2.19% pos) | rank |
|---|---|---|---|---|
| `qwen3.5-397b` | 0.821 | 1 | 0.790 | 2 |
| `gemma-4-31b` | 0.819 | 2 | **0.774** | **4** |
| `nemotron-3-super-120b` | 0.815 | 3 | **0.804** | **1** |
| `gpt-oss-20b` | 0.776 | 4 | 0.783 | 3 |

The order is scrambled: gemma goes 2nd → last, nemotron 3rd → 1st. The pilot's "**scale
saturates at ~31B**" was an artifact of the prevalence it was measured at — gemma was the
value pick there and is the weakest model here. This is `CONTEXT.md` §4's "model ranking is
not stable across prevalence regimes" reproduced on a second corpus, and it retires a
recommendation the pilot made.

## 3. 🟢 The F2-asymmetry sentence stops working at production prevalence

The pilot's most-promoted finding, and the one it said was "a property of prompted screening,
not of any model": adding *a missed relevant paper is about five times as costly as a false
positive; when uncertain, include* moved F2@own by **+0.31 to +0.37** with no labelled data.

At 2.19% each marginal inclusion costs ~45 false positives per true one. Re-priced:

| Model | F2@own P1 → P2 | recall | corpus read | AUC |
|---|---|---|---|---|
| `gemma-4-31b` | 0.333 → 0.293 (**−0.040**) | 0.388 → 0.486 | 0.069 → **0.148** | 0.759 → 0.774 |
| `gpt-oss-20b` | 0.382 → 0.402 (**+0.020**) | 0.434 → 0.490 | 0.086 → 0.103 | 0.772 → 0.783 |

The instruction still does what it says — recall rises on both models — but the price changed.
gemma more than doubles the pile a reviewer must read to gain 10pp of recall, which at this
prevalence is a net loss. **A +0.35 effect became ±0.03.** Ranking barely moves either way
(+0.015, +0.011), which is consistent with the pilot's own reading that the sentence relocates
the decision boundary rather than improving judgement — it is just that the boundary was in
roughly the right place to begin with here.

## 4. 🟢 The falsification control now passes on SYNERGY — closing an open risk

Every collection's papers re-scored against a **different** collection's brief (a derangement).

| | Real brief | Shuffled |
|---|---|---|
| mean AUC | 0.777 | **0.498** |
| F2@own | 0.285 | **0.000** |
| predicted-positive rate | 0.148 | **0.000** |
| WSS@95 | 0.331 | **−0.016** |

It does not degrade, it **refuses** — the model marks nothing relevant when the criteria belong
to someone else's question, which is the correct answer.

This matters beyond the control itself. `CONTEXT.md` §4 lists as an open risk that the
shuffled-brief control **fails on SYNERGY** (1/3, −0.012), diagnosing the cause as briefs that
are "a description of what a review did, not a statement of what to include" carrying "no
curated term lists". This export has term lists, the control now passes decisively, and §5
below shows the term lists are *not* what fixed it. **The open risk can be closed as a brief-
format artefact, not a property of SYNERGY.**

## 5. 🟢 The headline: what 60 labels buy, spent three ways

The induced rule set (B2, the "super use case") saw **30 positive + 30 negative train rows per
collection**. `CONTEXT.md` §1's rule — *cosine-to-brief below ~25 in-silo labels, supervised
in-silo model above* — puts 60 above the line, so **the cold-start cosine is not B2's fair
opponent.** A supervised model on exactly the same 60 rows is. All rows below are the same
3,997 held-out (test+validate) rows, all arms use prompt P2 and `gemma-4-31b`:

| Method | labels | mean AUC | F2@own | recall | corpus read |
|---|---|---|---|---|---|
| cosine-to-brief | 0 | 0.774 | — | — | — |
| LLM, supplied brief (B1) | 0 | 0.777 | 0.285 | 0.484 | 0.148 |
| LLM, raw review abstract (B0) | 0 | 0.777 | 0.349 | 0.559 | 0.147 |
| LogReg, query-conditioned block | 60 | 0.788 | 0.393 \* | 0.790 | 0.328 |
| **LLM, induced rule set (B2)** | 60 | **0.834** | **0.488** | 0.674 | **0.169** |
| LogReg, Qwen3-4B embedding | 60 | **0.844** | 0.413 \* | 0.790 | 0.303 |

\* A LogReg's operating point here is an artefact of `class_weight="balanced"` at p=0.5, not a
chosen threshold. **This is the asymmetry that matters: the LLM produces an operating point,
a ranker cannot.** Choosing a threshold for the LogReg needs labels held back for the purpose;
the LLM's 0.169 came out of the box.

Three readings, in decreasing confidence:

- 🟢 **Brief quality is the dominant lever, worth more than model choice.** B1 → B2 is
  **+0.057 AUC, +0.203 F2@own, +0.190 recall for +0.021 more reading** — 4 collections improved
  past the floor, 0 degraded. For comparison, the entire spread across four model families at
  20B–397B is 0.030. Same model, same prompt, better brief beats a 13× larger model.
- 🟢 **B2 ties a supervised embedding model on ranking and wins on work.** 0.834 vs 0.844 is
  inside the noise floor (2 wins, 2 losses, 4 ties per collection), but B2 reaches its recall
  reading **17% of the corpus against 30%**. Where a collection has ~60 labels, distilling them
  into a brief is competitive with training on them, and produces a deployable decision rather
  than a score needing a threshold.
- 🟢 **The supplied term lists are worth nothing, and cost recall.** B0 strips
  `terms_must_include` / `nice_to_have` / `exclude` back to the review's own title and
  abstract. Ranking is *identical* (0.777 both) and B0's operating point is **better** — recall
  0.559 vs 0.484 at the same reading cost. The LLM-written term lists make the model stricter
  without making it more accurate. Note this is the opposite of their effect on the lexical
  block, where curated terminology is load-bearing (`CONTEXT.md` §4): **term lists help
  keyword matchers and hurt readers.** Do not generalise "brief quality matters" into "add
  term lists".

The mechanism is visible per collection. On `leenaars_2020` the supplied brief drives gemma to
recall **0.043** (it rejects almost everything); the induced brief gets **0.713**. Same on
`muthu_2021`: 0.022 → 0.485. Two of eight collections were catastrophic under the shipped
brief, and both are fixed by naming the near-misses explicitly.

## 6. 🟢 `moran_2021`: some inclusion rules are learnable but not statable

One collection where a cosine scores **below chance** on 5,154 papers, and it is the most
informative row in the run. (All figures on the same held-out rows as §5; on the full 5,154 the
cosine is 0.445 and the picture is the same.)

| Method | AUC on `moran_2021` |
|---|---|
| cosine-to-brief | 0.487 |
| all four LLMs, supplied brief | 0.450 – 0.502 |
| LLM, raw brief (B0) | 0.488 |
| LLM, **induced brief from its own labels** (B2) | 0.462 |
| LogReg, query-conditioned block, 60 labels | 0.573 |
| **LogReg, Qwen3-4B embedding, 60 labels** | **0.639** |

A linear probe on the embedding finds **+0.152** over the cosine from 60 labels. No LLM
exceeds chance under any of four briefs — *including one distilled from those exact labels by
the strongest model available.* The induced brief for this collection is not vague; it names
the operative distinction explicitly ("excluding observational correlations, human clinical
trials, livestock management reports"). It still does not work.

So the signal is there and it is learnable — **it is just not expressible as a rule a screener
can apply.** That is a hard bound on the distil-labels-into-a-brief strategy, and it is the
reason §5's tie should not be read as "briefs replace training". Correcting an earlier reading
of my own: this is not a corpus defect. The labels are learnable; the *description* is what
fails.

## 7. ⚪ Engineering: throughput is a capability, and capability tables are claims

The pilot found provider choice moving measured compliance from 68% to 100%. 62,229 rows
exposes a failure 1,848 rows cannot: `nemotron-3-super` has three endpoints and all three fail
differently, none of it visible in OpenRouter's table.

| Endpoint | Price/M in | What actually happens |
|---|---|---|
| DeepInfra | $0.085 | Rejects `seed`. Rate-limits so hard at scale that measured throughput collapses to **4 rows/min** — 42 hours for one cell. Fine on 1,848 rows, unusable on 62,229. |
| DigitalOcean | $0.165 | **404s every request.** Advertises neither `temperature` nor `seed`, so `require_parameters` filters it to nothing. |
| Nebius | $0.300 | Works: 100% parse, 434 rows/min. Still rejects `seed` — **9,993 of 9,993 calls** had to drop it. |

Three consequences worth carrying forward:

- **The only usable endpoint costs 3.5× the cheapest and still cannot honour `seed`.** Against
  a "100% deterministic" product requirement, nemotron — the *best-ranking* model on set A —
  is disqualified on infrastructure, not on accuracy.
- **Throughput belongs in the capability check, not just latency and price.** DeepInfra's p50
  was a healthy 3.9s; the throughput was 1/100th of usable. Benchmark at the scale you intend
  to run at.
- **Four capability claims have now failed on contact** across this project (DeepInfra `seed`,
  Nebius `seed`, Alibaba `logprobs`, DigitalOcean `temperature`). Verify per endpoint, and pin.

Cheap news, unchanged: `gpt-oss-20b` scored 9,993 papers for **$0.42** — about **$4.20 per
100,000 papers**, at mean AUC 0.783, above the cosine. Cost is not the constraint at any scale
here; throughput and determinism are.

## 8. Negative results — do not re-run these

| Rejected | Evidence |
|---|---|
| **Zero-shot prompting as a replacement for the cold-start cosine** | 3/8 collections for every one of four models, 20B–397B. Bar was ≥6/8, fixed in advance |
| **Adding LLM-written term lists to a brief** | B0 = B1 on ranking (0.777) and B0 is *better* at the operating point (recall 0.559 vs 0.484). They add strictness, not accuracy |
| **"Scale saturates at ~31B"** (pilot claim, now retired) | gemma-31b is last of four on set A; nemotron-120b is first. The pilot's ranking was a prevalence artifact |
| **Relying on the F2-asymmetry sentence to set the operating point at production prevalence** | +0.35 F2@own on TIRI becomes −0.040/+0.020 here. Keep the sentence, do not rely on it |
| **`moran_2021`-style collections via any brief** | 4 briefs × 4 models, none above chance, while a 60-label linear probe reaches 0.639 |
| **DeepInfra for `nemotron-3-super` at corpus scale** | 4 rows/min measured. Nebius at 3.5× the price is the only working option |

## 9. Caveats

1. **Briefs sit closer to the answer than a real protocol.** Every brief here derives from the
   review's abstract — a summary of the papers that were included. This inflates absolute
   scores for *every* brief-reading method equally, cosine included. The dataset's own framing:
   fine for comparing methods, not proof of behaviour on a fresh question. B2 goes further and
   sees labels directly, which is why it is reported only on unseen rows and against
   same-label-budget baselines.
2. **B2 is not compared against few-shot.** It spends 60 labels on a brief; putting the same 60
   in the prompt as demonstrations (P3, never run) is the missing arm. §5's claim is that a
   brief is competitive with *training*, not that it is the best use of 60 labels.
3. **The ladder is one model.** B0/B2/shuffled ran on `gemma-4-31b` only, for budget. That
   gemma is the *weakest* zero-shot model here makes the +0.057 brief effect a conservative
   estimate rather than a flattering one, but it is unreplicated across families.
4. **All SYNERGY.** Eight SYNERGY collections and nothing else. §2 is the direct warning
   against assuming these results transfer to TIRI's own use cases — the pilot's did not
   transfer here.
5. **WSS@95 is soft for LLM arms.** Verbalised scores are lumpy (tie fraction 0.997–0.999), so
   the 95% cutoff falls inside a large tie block. `benchset_metrics.wss_at` resolves ties by
   expectation and reports `tie_frac` beside every ranking number, but (recall, fraction read)
   at the model's own verdict is the trustworthy operating-point comparison.
6. **Still no determinism result.** Stage C (repeat runs at temperature 0, flip rate) has never
   been run, and §7 shows the best model on this corpus cannot honour `seed` at all.
7. **Single seed for the sample.** The gate re-ran the *baseline* at three seeds (AUC sd 0.003);
   the LLM cells were bought once, on seed 0.

## 10. What this changes, and what to do next

**Closed by this run:** `CONTEXT.md` §4's shuffled-brief-fails-on-SYNERGY risk (§4 above);
the pilot's scale-saturation claim (§2); the pilot's §8 question about whether the benchset
would change the answer (it did — the ranking, the model choice, and the F2-asymmetry finding
all moved).

**The one thing worth building on:** brief quality, measured at +0.057 AUC / +0.203 F2@own for
the same model and prompt, is a larger and cheaper lever than model selection, and it applies
to the cold-start rung the product actually ships. The next three experiments, in order:

1. **B2 vs few-shot on the same 60 labels** (caveat 2). Cheap, and it decides whether the
   artefact should be a brief or a set of examples. A brief is far more attractive
   operationally — it is human-readable, auditable and editable by the analyst.
2. **Replicate the ladder on a second model family** (caveat 3), which is ~$5.
3. **Test the induced brief on the lexical/embedding block, not just the LLM.** If a better
   brief also lifts `cos_brief_*` and the BM25 features, the finding is about briefs and not
   about LLMs at all — which would be the most valuable version of it, and §5's B0 result
   (term lists help matchers, hurt readers) says the answer is not obvious.

**Not recommended:** further zero-shot model sweeps. Four families spanning 20B–397B land
within 0.030 of each other and all fail the same bar in the same 5 collections.
