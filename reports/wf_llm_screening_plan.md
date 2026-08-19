# LLM screening bake-off — test plan

**Status:** plan, nothing run yet. No API spend has occurred.
**Question:** can a prompted open-weights LLM beat the per-silo CatBoost+LogReg ensemble on F2?

**Confidence key:** 🟢 verified against data/API this session · 🟡 estimate · ⚪ open decision.

---

## 1. What we are trying to beat

Per-silo ensemble, within-silo held-out seeds (`wf_ensemble_v2_experiments.md` §9):

| | mean | range |
|---|---|---|
| ROC-AUC | 0.865 | 0.827 (soil) – 0.911 (cement) |
| F2@t\* | 0.893 | 0.726 (soil) – 0.950 (solar) |

🟢 **`f2_at_t_star` is an oracle number.** `f2_optimal()` sweeps 91 thresholds and keeps the
best, scored on the same OOF predictions it selected on. It is an upper bound, not a
deployable figure. Any LLM comparison must either give both sides the oracle, or give
neither — never one each.

**The strategic hypothesis.** The ensemble's threshold is tuned at 26–77% prevalence and
does not transfer downward; an LLM makes a criteria-based decision that should be
prevalence-invariant. So the interesting claim is not "LLM wins on TIRI's enriched pools"
(it probably won't) but "LLM holds its F2 where the ensemble's threshold stops
transferring." The benchset corpus (§4) is what tests that.

## 2. Two contests, not one

| | Contest | Incumbent | Surface |
|---|---|---|---|
| **(a)** | Beat the trained per-silo ensemble | CatBoost+LogReg, F2@t\* 0.893 | TIRI 6 use cases; benchset Tier M/L |
| **(b)** | Beat the **cold-start ranker** | cosine-to-brief (CONTEXT.md §1 ladder, 0–24 labels) | benchset Tier S — 11 collections too small to train on |

(b) is more likely to land and replaces a rung of shipped product rather than winning a
benchmark. It is a first-class objective, not a consolation prize.

## 3. Models — 23 slots, 0.8B → 550B, open-weights only

🟢 Verified against the live OpenRouter catalogue and HuggingFace this session.
CONTEXT.md §1 requires sovereign infrastructure, so **API-only weights are disqualified
regardless of score**: `qwen3.7-*`, `qwen3.5-flash/plus`, `qwen3.x-max` have no HF release.
This removes the cheapest model on the board (`qwen3.7-flash`, $0.030/M).

🟢 Every candidate supports `seed`; all but `ministral-14b` and `nemotron-3-ultra` support
`logprobs`. All but Ministral have reasoning modes that **must be explicitly disabled** or
output tokens and determinism both blow up.

### On OpenRouter (headline price; a pinned provider may cost more)

| # | Model | Size (active) | $/M in→out | Family |
|---|---|---|---|---|
| 1 | `qwen/qwen3.5-9b` | 9B dense | 0.100 → 0.150 | Qwen |
| 2 | `mistralai/ministral-14b-2512` | 14B dense | 0.200 → 0.200 | Mistral |
| 3 | `openai/gpt-oss-20b` | 20B MoE (~3.6B) | 0.030 → 0.130 | gpt-oss |
| 4 | `google/gemma-4-26b-a4b-it` | 26B MoE (4B) | 0.120 → 0.400 | Google |
| 5 | `qwen/qwen3.6-27b` | 27B dense | 0.600 → 3.600 | Qwen |
| 6 | `nvidia/nemotron-3-nano-30b-a3b` | 30B MoE (3B) | 0.050 → 0.200 | Nemotron |
| 7 | `nvidia/nemotron-3.5-lightning` | 30B-A3B | 0.100 → 0.250 | Nemotron |
| 8 | `google/gemma-4-31b-it` | 31B dense | 0.100 → 0.340 | Google |
| 9 | `qwen/qwen3.6-35b-a3b` | 35B MoE (3B) | 0.150 → 1.000 | Qwen |
| 10 | `openai/gpt-oss-120b` | 120B MoE (5B) | 0.037 → 0.170 | gpt-oss |
| 11 | `nvidia/nemotron-3-super-120b-a12b` | 120B MoE (12B) | 0.085 → 0.400 | Nemotron |
| 12 | `qwen/qwen3.5-122b-a10b` | 122B MoE (10B) | 0.290 → 2.400 | Qwen |
| 13 | `deepseek/deepseek-v4-flash` | large MoE | 0.140 → 0.280 | DeepSeek |
| 14 | `z-ai/glm-5.2` | large MoE | 0.420 → 1.400 | Zhipu |
| 15 | `qwen/qwen3.5-397b-a17b` | 397B MoE (17B) | 0.500 → 3.600 | Qwen |
| 16 | `nvidia/nemotron-3-ultra-550b-a55b` | 550B MoE (55B) | 0.600 → 3.600 | Nemotron |

Slot 7 shipped 2026-08-11 — 2 providers, ~16k HF downloads. In, with a stability caveat.

### Below OpenRouter's floor → Modal + HF

Nothing under ~9B is served for these families, and the sub-5B tier is where the
"runs on our own hardware for pennies" argument lives. One vLLM app,
`scripts/modal_llm_screening.py` — all LLM Modal functions in that one file
(CONTEXT.md §7 is explicit about not splitting them).

| # | HF model | Size |
|---|---|---|
| M1 | `Qwen/Qwen3.5-0.8B` | 0.8B |
| M2 | `google/gemma-4-E2B-it` | ~2B eff. |
| M3 | `Qwen/Qwen3.5-2B` | 2B |
| M4 | `nvidia/NVIDIA-Nemotron-3-Nano-4B` | 4B |
| M5 | `Qwen/Qwen3.5-4B` | 4B |
| M6 | `google/gemma-4-E4B-it` | ~4B eff. |
| M7 | `google/gemma-4-12B-it` | 12B |

## 4. The benchset corpus

🟢 On disk at `<academic_agent>/exports/benchsets_v1/`.
181,199 rows · 3,374 positive (1.86%) · 28 collections. Read-only from TIRI sessions.

### 4.1 Quarantine — three collections are already burned

🟢 `synergy_sep_2021`, `synergy_menon_2022` and `synergy_van_der_waal_2022` are all three
present, and all three already drove live decisions in this repo: the Jasper+Qwen3-8B
embedding swap (recommendation #4) and the `C≈0.0005` cold-start LogReg default (#9).
They are not clean holdout — they are the selection-on-holdout hazard CONTEXT.md §4 names.

Removing them costs **3,216 rows (1.8%) and 147 positives (4.4%)** — and *improves* the
corpus, because `sep_2021` (14.8%) and `menon_2022` (7.6%) are high-prevalence outliers.
⚪ Decision: quarantine, report separately, never in a headline mean.

### 4.2 Tiering — 🟢 computed from `manifest.json`

| Tier | Collections | Rows | Positives | Role |
|---|---|---|---|---|
| **S** small | 11 (<40 pos) | 30,608 | 233 | Contest (b) — cold-start |
| **M** mid | 11 | 60,786 | 2,069 | Contest (a) — main surface |
| **L** giants | `walker_2018`, `brouwer_2019` | 86,457 | 822 | Winner-only confirmation |
| — burned | 3 | 3,216 | 147 | Quarantined |
| — excluded | `roadfreight_metareview` (78% pos, selects reviews) | 132 | 103 | Never in headline |

**Tier S+M = 91,394 rows is the working surface.** Tier L is half the corpus by rows and
adds little per dollar — except `brouwer_2019`, which at **0.16% prevalence (62 positives
in 38,114)** is the single most production-realistic collection in the set and must be run.

### 4.3 What this corpus replaces

Prevalence spans **0.16% → 21.9%** across 27 usable collections. That is a real
prevalence-transfer curve — metric plotted against prevalence, 27 points. It is strictly
better than the synthetic positive-downsampling stress test previously proposed, which is
**dropped**.

### 4.4 The synthetic-brief caveat

Briefs are LLM-drafted from each review's title+abstract, blind to labels
(`brief_provenance` records this per row). Two consequences:

- **Good:** they carry `terms_must_include` / `nice_to_have` / `terms_exclude`, matching
  TIRI's own schema. `build_use_case_brief` and the `brief_map=` swap seam work unchanged,
  and `briefs.parquet` makes brief-swapping a first-class operation. The format defect that
  made the shuffled-brief control *fail* on old SYNERGY is gone.
- **Cautionary:** written from a summary of the papers that got included, they sit closer
  to the answer than a real pre-screening protocol. This inflates absolute scores for every
  brief-reading method equally. Adopt the dataset README's own framing verbatim: *fine for
  comparing methods against each other; not proof of what a system would do on a fresh
  question.*

## 5. Prompt variants

| | Variant | Factor isolated |
|---|---|---|
| P0 | Bare zero-shot binary | Floor |
| P1 | Structured brief + 0–100 score | Graded score vs. label; enables ranking metrics |
| P2 | P1 + explicit F2 asymmetry ("a miss costs ~5× a false positive") | Most likely direct F2 lever; untested in this repo |
| P3 | P1 + few-shot k=8, **train-fold only** | Do in-context labels close the gap? |
| P4 | P1 + per-criterion checklist before verdict | Reasoning scaffold; yields per-score notes free |
| P5 | **Induced "super use case" rule set, train-fold only** | Is the brief the bottleneck? |
| P1-lp | P1 scored by **logprob of the "yes" token** | Continuous score; verbalised confidence clusters on 70/80/90 |

**P5 is a fitted parameter, not a prompt.** Induced from training-fold rows only and
re-induced per fold — the same rule `select_few_shot_examples` already enforces for few-shot
demonstrations, and for the same reason. It must be versioned and pinned alongside model and
prompt version; `relevance_score` was disqualified as a feature for exactly this drift
(`wf_ensemble_report.md` §0).

🟡 **KV caching for P5:** provider-specific, measured live — DeepSeek-V4-Flash $0.090 →
$0.018 (5×), GLM-5.2 $0.42 → $0.07 (6×), Gemma-4-31B/Chutes $0.12 → $0.012 (10×). Immaterial
at test scale; material at 100k-paper production scale. Note the tension: cache support is
per-provider and the cheapest provider often has none, so pinning for determinism can cost
the discount.

## 6. Controls

- **Shuffled brief** — the repo's signature falsification test, now first-class via
  `briefs.parquet`. Runs on all rows. **Primary control.**
- **Brief-format ladder** — raw review title+abstract → supplied AI brief → induced rule set.
  Three points, one measurement of what brief format is worth. Answers the question
  CONTEXT.md §4 flags twice and nobody has quantified.
- **Cross-collection rule set** — induce from collection A's labels, apply to B. If it still
  scores, it encodes generic "good paper" heuristics.
- 🟢 **Same-paper-different-label natural experiment** — 3,137 papers appear in >1 collection;
  **130 of them carry conflicting human labels** (265 rows). Real counterfactual, not
  synthetic. **Power is limited** — ~4.4pp binomial SE, and 65 of the 130 come from a single
  pair (`hall_2012` + `radjenovic_2013`), so it is one domain more than four. Supporting
  diagnostic, not a headline statistic. The 3,007 concordant duplicates test verdict
  stability under a changed brief.
- **Abstention path** — `appenzeller-herzog_2019` (77% abstract coverage) and `chou_2004`
  (88%) are the natural test beds. Nulls stay null; NULL ≠ 0.

## 7. Staging and budget

Promotion thresholds are **pre-registered before any run**. With ~23 slots × 7 variants the
multiple-comparisons hazard is real, and CONTEXT.md §5's measured noise floor (~0.010 AUC
seed-to-seed; treat <0.03 as not established) is the bar. Stage A's own threshold comes from
a bootstrap of its 360-row subsample noise, not from eyeballing a top-5.

| Stage | Surface | 🟡 Cost |
|---|---|---|
| A | 16 OR slots × P1 × 360-row TIRI subsample. Screens rank **and** JSON compliance | ~$3 |
| A2 | 7 Modal slots, same rows, one vLLM app | ~$3 credits |
| B | 5 finalists × 7 variants × TIRI 1,848 | ~$26 |
| C | Top 3 cells × 3 repeats, temp 0, provider pinned — flip rate + F2 spread | ~$8 |
| D1 | 5 finalists × 2 best variants × benchset Tier S+M (91,394 rows) | ~$75 |
| D2 | Winner + cheap floor × Tier L (incl. `brouwer_2019` at 0.16%) | ~$15 |
| D3 | Controls (§6) | ~$15 |
| E | Blend analysis — no spend; tuned CatBoost OOF already on disk and row-aligned | — |

**~$140–160 OpenRouter + ~$10 Modal credits.**

🟢 **Wall clock is the binding constraint, not money.** One Tier S+M pass is 91,394 calls
≈ 2.6 h at 24-way concurrency, ≈ 40 min at 100-way. Ten cells is an overnight run. Different
models on different providers have independent rate limits, so cells should run in parallel.

## 8. Build

- `scripts/llm_pipeline_utils.py` — replace `call_llm_stub` with a real async OpenRouter
  client. On-disk response cache keyed by `hash(corpus, model, prompt, params)` so
  re-analysis is free and every number is auditable. Log `model`, `prompt_version`,
  `rule_set_version` and the response's `provider` on every row.
- **Corpus loader is parameterised from the start** — data path, label column, brief columns,
  group column. TIRI and benchset differ only by config.
- `scripts/run_llm_screening.py` — stage runner. Reuses `ensemble_eval_utils.scores` and
  `f2_optimal`; no metric re-implemented.
- `scripts/modal_llm_screening.py` — vLLM app for the 7 sub-9B slots.
- `notebooks/main/10_llm_screening.ipynb` — stage-by-stage diagnostic: model×prompt heatmap,
  per-collection F2, **metric vs. prevalence across 27 collections**, branch correlation
  against the ensemble OOF, confidence calibration, brief-format ladder.

## 9. Adopt / reject bar — fixed before running

- **Replaces the ensemble** if it beats 0.893 F2@t\* on TIRI *and* wins on benchset Tier M,
  by >0.03, on ≥⅔ of collections, with the shuffled-brief control passing.
- **Replaces the cold-start rung** if it beats cosine-to-brief on Tier S by >0.03. Independent
  of the above, and the more likely win.
- **Joins as a third ensemble branch** if ρ with both existing branches is <0.7 and nested
  3-way blending gains >0.03 F2 on held-out seeds. The 3-lever sweep rejected SVM, lexical-only
  and k-NN for being too correlated (ρ 0.74–0.91); an LLM is the first candidate with a
  genuinely different *feature view*.
- **Rejected but documented** otherwise — into CONTEXT.md §6's negative-results register with
  the shuffled-brief number attached, so nobody re-runs it.

## 10. Open decisions

- ⚪ Quarantine the 3 burned collections (§4.1)? Recommended; costs 1.8% of rows.
- ⚪ Tier L scope — `brouwer_2019` is essential; `walker_2018` (48,343 rows) is the single
  most expensive collection and adds a 1.57%-prevalence point we already have neighbours for.
- ⚪ Whether contest (b) alone justifies shipping, if (a) fails.
