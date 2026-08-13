# LLM screening pilot — findings

> **Three claims below have since been overtaken — see `wf_llm_benchset_a_findings.md` and
> `wf_llm_logprob_scoring.md`.** The measurements stand; the conclusions drawn from them do
> not, because this pilot ran at 26–77% prevalence and set A ran the same models at 2.19%.
> Specifically: §5's **"scale saturates at ~31B"** is a prevalence artefact (`gemma-4-31b` is
> *last* of four on set A, `nemotron-120b` *first*); §2's **F2-asymmetry gain of +0.31 to
> +0.37** shrinks to −0.040/+0.020 at production prevalence, because "when uncertain, include"
> stops being cheap; and §6's recommendation to **use token logprobs for ranking** was tested
> and rejected — it fixed the tie fraction (0.997 → 0.047) and made ranking *worse* (mean AUC
> −0.035), so the 0.044 ranking gap in §1 is genuine and not an elicitation artefact.
> Everything else here, including §3's control and §4's third-branch rejection, is unaffected.

**What this is:** the decision doc for the pilot specified in `wf_llm_screening_plan.md` —
can a prompted open-weights LLM beat the per-silo CatBoost+LogReg ensemble on F2?

**Answer: no, it does not replace the ensemble. Two of the things it did instead are worth
keeping, and one of them does not depend on LLMs at all.**

**Scope, stated up front:** TIRI's own 1,848 labelled rows across 6 use cases, at 26–77%
prevalence. `CONTEXT.md` §4 says these pools are "a poor surface for judging a
recall-oriented system." So this pilot **could have killed the idea cheaply and did not;
it cannot confirm it.** Nothing here is a production estimate.

**Cost:** $4.90 for the 12-cell grid (22,176 responses, 99.98% parsed) + $0.81 for the
shuffled-brief control. Stage E cost nothing — the CatBoost OOF was already on disk.

**Confidence key:** 🟢 clears the noise floor / decisively measured · 🟡 real but under the
floor · ⚪ engineering finding.

---

## 1. The bar, and how it went

`f2_at_t_star` is an **oracle**: `f2_optimal` sweeps 91 thresholds and keeps the best on
the same predictions it selected on. It is reported for both sides or neither. `f2_at_own`
is the LLM's own verdict with nothing fitted — the honest operating point, and the one that
should survive a prevalence shift when a swept threshold does not.

| | ROC-AUC | F2@t* (oracle) | F2@own (honest) | ρ vs CatBoost |
|---|---|---|---|---|
| **Ensemble (incumbent)** | **0.865** | **0.893** | — | — |
| `qwen/qwen3.5-397b-a17b` P2 | 0.821 | 0.881 | 0.812 | 0.628 |
| `google/gemma-4-31b-it` P2 | 0.819 | 0.875 | **0.814** | 0.605 |
| `nvidia/nemotron-3-super-120b` P2 | 0.815 | 0.880 | 0.648 | 0.614 |
| `openai/gpt-oss-20b` P2 | 0.776 | 0.861 | 0.541 | 0.512 |

| Stop condition (pre-registered) | Result |
|---|---|
| Ranking — beat 0.865, else stop below 0.75 | 0.821. Above the floor, below the ensemble by 0.044 |
| Scale — is bigger better? | +0.045 from 20B to 397B, but **saturated by 31B** |
| Decorrelation — ρ < 0.7 ⇒ third-branch case | **0.51–0.63 on all 12 cells.** Passes |
| F2@t* vs 0.893 | −0.009. A tie on the oracle metric |

**Verdict: does not replace the ensemble.** It loses on ranking by more than the noise
floor and ties on oracle-F2.

## 2. 🟢 The finding that does not depend on LLMs: state the F2 asymmetry

P2 is P1 plus one sentence — *"a missed relevant paper is about five times as costly as a
false positive; when uncertain, include."* Effect:

| P1 → P2 | F2@own (operating point) | F2@t* (ranking ceiling) |
|---|---|---|
| `gemma-4-31b` | 0.507 → **0.814** (+0.307) | 0.874 → 0.875 (+0.001) |
| `qwen3.5-397b` | 0.445 → **0.812** (+0.367) | 0.880 → 0.881 (+0.001) |
| `gpt-oss-20b` | 0.428 → 0.541 (+0.113) | 0.824 → 0.861 (+0.037) |
| `nemotron-120b` | 0.608 → 0.648 (+0.040) | 0.870 → 0.880 (+0.010) |

**The ranking does not move; the operating point moves enormously.** That is the signature
of a model that already knew the answer and was drawing its line in the wrong place. It
closes ~84% of the gap to the oracle threshold **with no labelled data to tune against** —
precisely what a swept threshold cannot do at a prevalence it was not tuned at.

This is a property of prompted screening, not of any model here, and it addresses the
cold-start rung of `CONTEXT.md` §1's ladder directly.

## 3. 🟢 The control: the LLM really is reading the brief

Every use case's papers re-scored against a **different** use case's brief (a derangement).

| | Real brief | Shuffled brief |
|---|---|---|
| `gemma-4-31b` P2 — AUC | 0.819 | **0.488** |
| `qwen3.5-397b` P2 — AUC | 0.821 | **0.496** |
| F2@own | ~0.81 | **0.000** |
| Use cases degraded | — | **6/6** |

It does not merely degrade — it **refuses**: 1,764 of 1,848 papers scored exactly 0.00,
predicted-positive rate 0.0. AUC of 0.500 here is a *constant* score, not a weak ranking.
The models are saying "none of these match these criteria," which is correct when the
criteria belong to someone else's question.

Sharper than the lexical block's own pass on this control (5/6, +0.155). This is what makes
§1 and §4 interpretable at all.

## 4. 🟡 As a third branch: the best candidate yet tested, still under the floor

| Third branch | ρ vs CatBoost | mean AUC gain | mean F2@t* gain | Verdict |
|---|---|---|---|---|
| SVM (rbf) | 0.86–0.91 | +0.010 | negative | rejected |
| lexical-only | 0.52–0.67 | +0.006 | −0.003 | rejected |
| k-NN | 0.74–0.87 | +0.003 | −0.005 | rejected |
| `gpt-oss-20b` P2 | **0.512** | +0.010 | +0.006 | reject |
| `gemma-4-31b` P2 | 0.605 | +0.017 | +0.004 | reject |
| `qwen3.5-397b` P2 | 0.628 | **+0.019** | **+0.005** | reject |

Roughly double SVM's AUC gain and the only candidate of the four with a **positive** F2
gain — but the floor is 0.03 and nothing reaches it.

**The mechanism, now visible within one branch across use cases** (`gemma`):

| Use case | ρ | AUC gain |
|---|---|---|
| Carbon Capture | 0.699 | +0.034 |
| Named Entity Recognition | 0.651 | +0.034 |
| Technology Prediction | 0.587 | +0.024 |
| Solar Cells | 0.491 | +0.007 |
| Soil Microbiome | **0.404** | **−0.004** |

**The LLM helps where it is strong, not where it is different.** Soil and Solar are
simultaneously the most decorrelated and the least useful — and are exactly where its
standalone AUC is worst (0.689, 0.739). Same mechanism that killed the lexical branch,
reproduced inside a single branch. Sharper statement of the rule than the three prior
rejections gave: **diversity only pays where the dissenting branch is competent.**

Secondary: the weight search consistently lands near **(0.5 CatBoost, 0.1 LogReg, 0.4 LLM)**
— it does not add the LLM alongside LogReg, it *displaces* it.

## 5. ⚪ Engineering findings worth more than they look

- **Provider choice moved measured "model compliance" from 68% to 100%** with model,
  prompt and parameters identical. `gemma` on Chutes returned 586 HTTP 504s and 68%
  coverage; on Friendli, 0 errors and 100%. Anyone benchmarking open-weights models through
  a routing layer without pinning is partly measuring their router. Latency spread across
  providers for one model is ~5x — far wider than the price spread.
- **`nemotron-3-super` cannot honour `seed` on any provider that advertises it.** Both
  reject it at the endpoint. Against a "100% deterministic" product requirement that is a
  deployability finding about the model.
- **Scale saturates at ~31B.** `gemma-4-31b` (0.819) ≈ `qwen3.5-397b` (0.821) at **1/3 the
  cost** ($0.198 vs $0.620 per 1,848 rows). By *active* parameters (3.6B / 31B / 17B) the
  story is active capacity, not size.
- **Cost is not the constraint.** `gpt-oss-20b` scores 1,848 papers for $0.065 including
  mandatory reasoning tokens — roughly **$4 per 100,000 papers**. The "LLMs are too
  expensive at pool scale" assumption is dead for small models.

## 6. Negative results — do not re-run these

| Rejected | Evidence |
|---|---|
| **P4, the per-criterion checklist** | Worse AUC than P1/P2 on 3 of 4 models and much worse F2@own (0.24–0.50). Working through criteria makes models demand *all* of them: predicted-positive rate collapses to 0.14–0.34. Reasoning scaffolding hurts a recall-oriented screen |
| **`gpt-oss-20b` as the shippable model** | Most decorrelated (ρ 0.512) but weakest (AUC 0.776, F2@own 0.541). Diversity without competence, again |
| **Verbalised 0–100 confidence as a ranking signal** | Tie fraction **>0.99** — 9–16 distinct values across 1,848 rows. ROC-AUC is computed over ~10 buckets against the ensemble's continuous probabilities, so part of the 0.044 ranking gap is elicitation, not judgement. Use token logprobs instead if ranking is the deciding metric |

## 7. Caveats

1. **Prevalence.** 26–77% positive vs production's low single digits. Rankings are more
   stable than thresholds across that shift, but neither is validated here.
2. **`solar_leo` flatters the ensemble.** Its labels track publication year (`CONTEXT.md`:
   corpus defect). The ensemble has `year`/`paper_age`; the LLM never sees them. Excluding
   it, the AUC gap narrows from 0.046 to **0.027 — inside the noise floor**. The LLM's
   worse score there is arguably the more honest one.
3. **Stage E is a seed-0 screening run**, not the 5-seed Modal protocol the SVM/lexical/
   k-NN experiments used. Weights were chosen leave-one-use-case-out rather than
   leave-seeds-out, deliberately: an LLM branch has no seed, so a weight tuned to how it
   fits these rows would carry straight into validation and a seed split would never catch
   it. The result is ~half the noise floor — not promising enough to buy the escalation.
4. **Single run, no determinism repeats.** Stage C (3 repeats at temp 0, flip rate) was not
   reached. Against a "100% deterministic" requirement this is still open — and §5's `seed`
   finding says at least one model cannot satisfy it at all.
5. **Selection-on-holdout.** Model and prompt were chosen against these same six use cases.

## 8. What would change the answer

- **Token-logprob scoring** instead of a verbalised 0–100. The >0.99 tie fraction is a
  measurement artifact handicapping the LLM in the one comparison it loses.
- **The benchset corpus** (28 collections, 1.86% prevalence) — the surface this pilot
  explicitly cannot substitute for. Two contests there: beat the ensemble where it can be
  trained, and beat the **cold-start cosine-to-brief ranker** on the 12 collections too
  small to train one. The second is more winnable and replaces a shipped rung.
- **P5, the induced "super use case" rule set** — untested here. §3 shows brief quality is
  load-bearing, which is the premise P5 rests on.
