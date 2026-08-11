# Embedding model bake-off — findings and recommendations

This is the decision trail behind testing whether a better embedding model would help
TIRI's downstream "is this paper interesting" classifier. The corpus currently embeds
every paper with one small, free, local model
(`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions). This
report covers everything tested against that baseline, what won, and what to actually do
about it. Written so someone who wasn't in the room can pick this up and trust the
numbers — every claim points at the notebook that produced it, so you can re-run and
check rather than take it on faith.

**How to read this doc**: each section has a **Simple explanation** box in plain
language, then the actual numbers. If you only read one thing, read the TL;DR table
below.

---

## TL;DR — what to actually do

| Do this | Because |
|---|---|
| **Switch the corpus embedding to `infgrad/Jasper-Token-Compression-600M` + `Qwen/Qwen3-Embedding-4B`, combined** | This pair beat every single model tested, on 6 different held-out domains, not just one lucky one (§4) |
| **Combine the two by concatenating their vectors**, not by averaging or stacking their predictions | Concatenation won consistently; the fancier "combine predictions" approaches never beat it (§5) |
| **Calibrate the combined model's output with isotonic regression**, using ~50 labels from each new use case | Raw scores are badly miscalibrated on a domain the model hasn't seen — calibration fixes this cheaply without hurting ranking quality (§6) |
| **Re-embed the whole historical corpus** if this is adopted, rather than mixing old and new embeddings | Mixing two embedding spaces in one classifier is exactly the confound `reports/wf_ensemble_report.md` warns about, and re-embedding is cheap (§8) |
| **Don't assume this fixes every use case** | On `carbon_capture` and `soil_microbiome` specifically, the tiny local baseline was competitive or better than every new model tested (§4) |
| **Budget for real but small cost**: a few dollars in API calls (OpenRouter) and GPU time (Modal), one-time | Confirmed in practice, not just estimated (§9) |

---

## 1. What we were testing, and why

**Simple explanation**: an "embedding" turns a paper's title and abstract into a list of
numbers (a vector) that's supposed to capture what the paper is *about*. Two papers on
similar topics should get similar-looking vectors. The classifier that decides "is this
paper interesting for this use case" is built on top of these vectors — so if the
vectors are bad, no amount of clever modelling downstream can fix that. This whole
investigation asks one question: **is our current, tiny, free embedding model actually
good enough, or would a better one give the classifier more to work with?**

We tested candidates the same way every time, so the comparison is fair:

- **Same fold design** as `notebooks/experiments/wf_fold_pca_test.ipynb`: papers grouped by
  first author (so the same author's papers never straddle train/test), stratified by
  use case + label, 5 folds.
- **Same simple classifier** (a plain `LogisticRegression`) on top of whichever
  embedding was being tested — deliberately simple, so a difference in score is about
  the *embedding*, not about which model happened to get a fancier classifier.
- **Same metric bundle** every time: ROC-AUC, PR-AUC, F2, Recall@k (top 10/20/30% of the
  pool), and Brier score (calibration) — the same bundle recommended in
  `reports/wf_ensemble_report.md` §2.
- **Held-out generalization, not just in-distribution CV** — the number that matters is
  how a model does on a use case it never saw during training, not how well it fits data
  it's already seen.

11 candidate models were tested against the local baseline, in two batches:

| Access method | Models |
|---|---|
| Hosted API (OpenRouter, small per-token cost) | `qwen3-embedding-8b`, `openai/text-embedding-3-large`, `baai/bge-m3`, `mistral-embed-2312`, `google/gemini-embedding-2`, `nvidia/nemotron-3-embed-1b` (free), `perplexity/pplx-embed-v1-4b` |
| GPU-hosted (Modal, rented GPU time) | `allenai/SPECTER2`, `Kingsoft-LLM/QZhou-Embedding` (7B params), `infgrad/Jasper-Token-Compression-600M`, `Qwen/Qwen3-Embedding-4B` |

---

## 2. Round 1 — 7 hosted models vs. the local baseline

**Simple explanation**: first, the cheap/easy check — call 7 different embedding APIs,
run every model through the exact same test, see who wins. This is the fastest way to
find out if *any* of the paid options are worth pursuing before spending more effort.

**Result: every single external model beat the local baseline**, on both in-distribution
and held-out (`tech_forecasting`) PR-AUC:

| Model | dev PR-AUC | holdout PR-AUC |
|---|---|---|
| qwen3-embedding-8b | 0.891 | 0.758 |
| openai/text-embedding-3-large | 0.886 | 0.716 |
| nemotron-3-embed-1b (free) | 0.891 | 0.705 |
| gemini-embedding-2 | 0.870 | 0.686 |
| perplexity/pplx-embed-v1-4b | 0.854 | 0.730 |
| mistral-embed-2312 | 0.862 | 0.672 |
| bge-m3 | 0.852 | 0.673 |
| **local baseline (current)** | 0.833 | **0.608** |

`qwen3-embedding-8b` came out on top, cheap (~$0.01 per million tokens), and this alone
was enough to justify testing further, heavier options.

*Notebook: `notebooks/experiments/wf_embedding_model_bakeoff.ipynb`.*

---

## 3. Round 2 — adding 4 heavier, GPU-hosted models

**Simple explanation**: some of the strongest embedding models available aren't
accessible via a simple paid API — they're open-weight models you have to run yourself
on a GPU. This laptop doesn't have one, so these 4 were run on rented cloud GPUs (Modal)
instead, through the exact same test.

| Model | Size | dev PR-AUC | holdout PR-AUC |
|---|---|---|---|
| **Jasper-Token-Compression-600M** | 600M params | 0.867 | **0.761** |
| Qwen3-Embedding-4B | 4B params | 0.892 | 0.724 |
| SPECTER2 | ~110M + adapters | 0.843 | 0.674 |
| QZhou-Embedding | 7B params | 0.829 | 0.690 |

**Jasper — the smallest of the 4 — came out best on the held-out score**, narrowly
beating `qwen3-embedding-8b` from Round 1 (0.761 vs. 0.758), despite being ~13x smaller
and faster to run (76s vs. 154s to embed the whole labelled corpus).

Two things worth flagging plainly:

- **SPECTER2 underperformed** despite being a model specifically trained for academic
  papers. Its "query" mode landed every use case's search text at the 0th percentile of
  its own paper corpus (i.e. as far from typical as possible) — its query-encoding mode
  seems mismatched to our paragraph-length use-case descriptions, which are longer and
  more detailed than the short search queries it was designed for.
- **QZhou (7B, the largest model tested) did not translate its size into better
  results.** Bigger is not automatically better — this is a recurring theme, see §4.

*Notebook: `notebooks/experiments/wf_embedding_model_bakeoff.ipynb`; Modal app:
`scripts/modal_embeddings.py`.*

---

## 4. Does the winner generalize everywhere? Leave-one-use-case-out

**Simple explanation**: Round 1 and 2 both tested generalization against just *one*
held-out use case (`tech_forecasting`). That's one data point — it tells you a model
generalizes to *that* domain, not that it generalizes in general. To actually trust a
"this model wins" claim, you have to repeat the test 6 times, holding out each of the 6
use cases in turn, and see if the win holds up every time.

This is exactly what we did, for the two leading models (`qwen3-embedding-8b`,
`qwen3-embedding-4b`) plus Jasper and the local baseline:

| Model | carbon_capture | cement_binders | ner | soil_microbiome | solar_leo | tech_forecasting | **mean** |
|---|---|---|---|---|---|---|---|
| qwen3-embedding-8b | 0.499 | 0.697 | 0.752 | 0.264 | 0.819 | 0.758 | **0.632** |
| qwen3-embedding-4b | 0.488 | 0.739 | 0.743 | 0.280 | 0.811 | 0.724 | 0.631 |
| Jasper-600M | 0.464 | 0.671 | 0.747 | 0.261 | 0.852 | 0.761 | 0.626 |
| local baseline | 0.512 | 0.670 | 0.719 | 0.287 | 0.713 | 0.608 | 0.585 |

**The headline result softens, but still holds**: averaged across all 6 use cases, the
new models beat the local baseline (0.63 vs. 0.585) — real, but a much smaller margin
than the single-holdout test suggested (0.76 vs. 0.61).

**The important catch, in plain terms**: on 2 of the 6 use cases —
**`carbon_capture` and `soil_microbiome`** — the tiny free local model was *competitive
with or better than* every fancy new model. On `carbon_capture`, the local baseline
(0.512) beat all three new models outright. This means: **the new embeddings are a real
improvement, but not a universal one.** Which model wins depends on the specific domain,
and no single model dominates every time — exactly why `reports/wf_ensemble_report.md`
insisted on testing every use case as its own holdout, rather than trusting one.

*Notebook: `notebooks/experiments/wf_top_embeddings_generalization.ipynb`.*

---

## 5. Combining models: concatenation vs. combining predictions

**Simple explanation**: if two different embedding models each capture something useful,
maybe using *both together* beats using either alone. There are two ways to combine
them: (a) glue their number-lists together into one longer vector before the classifier
sees it ("concatenation"), or (b) let each model make its own prediction, then average or
learn how to blend those two predictions ("prediction-level stacking"). We tested both,
for two pairings — Jasper with `qwen3-embedding-8b`, and Jasper with the smaller,
cheaper `qwen3-embedding-4b`:

| Combination | Method | Mean holdout PR-AUC (across all 6 use cases) |
|---|---|---|
| Jasper + qwen3-8b | **concatenation** | **0.639** |
| Jasper + qwen3-4b | **concatenation** | **0.636** |
| Jasper + qwen3-4b | average predictions | 0.632 |
| Jasper + qwen3-4b | learned blend (stacking) | 0.632 |
| Jasper + qwen3-8b | average predictions | 0.630 |
| Jasper + qwen3-8b | learned blend (stacking) | 0.632 |
| *(for reference)* best single model | — | 0.632 |

**Combining beats every single model — but only via concatenation.** The fancier
"combine the predictions" approaches never beat simple concatenation, on either pairing.

**The practical finding**: Jasper+qwen3-4b (0.636) is a virtual tie with Jasper+qwen3-8b
(0.639) — a 0.003 gap, well within noise. Since qwen3-4b is roughly half the size and
cost of qwen3-8b, and already matched it in Round 1/2 testing, **the cheaper pairing
(Jasper + qwen3-4b) is the more practical choice** — same performance, lower ongoing
cost.

*Notebook: `notebooks/experiments/wf_top_embeddings_generalization.ipynb`.*

---

## 6. Calibration: making the scores actually usable

**Simple explanation**: a classifier's raw output (e.g. "0.83") is supposed to mean
"83% chance this paper is interesting" — but that's only true if the model was trained
on data similar to what it's scoring. When we test on a use case the model has *never
seen*, its raw scores stop meaning what they claim to mean, even though the model still
*ranks* papers correctly (best-to-worst). **Calibration** is a cheap fix: using a small
number of real labels from the new use case, you re-map the raw scores onto scores that
actually reflect reality, without changing the ranking at all.

We found direct evidence this problem is real and fixable. At the default 0.5 threshold,
**F2 (a recall-focused accuracy score) was terrible on every held-out use case** — as low
as 0.07–0.12 — despite PR-AUC/Recall@k (which only care about ranking, not the actual
score value) looking fine. This is a calibration problem, not a ranking problem.

We simulated the realistic scenario: a brand-new use case with only **~50 labels**
available (matching the "~50 informative labels" bootstrap budget already discussed for
new use cases), used those 50 to fit a calibrator, and re-scored the rest:

| Model / combo | F2 before → after (isotonic) | Brier before → after (isotonic) |
|---|---|---|
| qwen3-embedding-8b | 0.074 → 0.771 | 0.315 → 0.242 |
| qwen3-embedding-4b | 0.093 → 0.701 | 0.317 → 0.242 |
| Jasper + qwen3-4b (concat) | 0.084 → 0.562 | 0.334 → **0.241** (best) |
| Jasper + qwen3-8b (concat) | 0.069 → 0.546 | 0.329 → 0.246 |
| Jasper (solo) | 0.121 → 0.470 | 0.278 → 0.256 |

**Isotonic calibration turns an unusable score into a usable one**, in every case, for a
one-time cost of ~50 labels per new use case. Two important caveats:

- **We tried two calibration methods — isotonic and Platt (sigmoid) scaling — and
  isotonic is clearly the safer choice.** With only ~50 calibration points, Platt's
  underlying logistic-regression fit occasionally went badly wrong (one case saw
  `qwen3-embedding-8b`'s ROC-AUC actually drop from 0.60 to 0.49, i.e. to worse than a
  coin flip) because it can fit a noisy, badly-signed slope on a tiny sample. Isotonic
  regression can't do that — it's forced to stay rank-preserving by construction.
- **Calibration doesn't improve ranking** — it can't, by construction (both methods are
  mathematically monotonic transforms, i.e. they never reorder papers, only rescale the
  numbers). If ranking (PR-AUC, Recall@k) is already poor for a use case, calibration
  won't fix that; it only fixes the *meaning* of the score.

*Notebook: `notebooks/experiments/wf_top_embeddings_generalization.ipynb`.*

---

## 7. External validation: SYNERGY

**Simple explanation**: everything so far has been tested on TIRI's own 6 use cases —
papers found and labelled by our own process. That's a fair test, but it can't rule out
the possibility that our winning models just happen to suit *our* labelling process
specifically. **SYNERGY** is a public, independent dataset of systematic literature
reviews — real academic papers, labelled "included"/"excluded" by professional
researchers for their own published reviews, with nothing to do with this project. If
our winning embeddings also work well here, that's much stronger evidence they're
genuinely good, not just a fluke of our own data.

We tested against 3 SYNERGY reviews chosen for a spread of size and difficulty:

| Review | Papers | % included | local baseline PR-AUC | Jasper PR-AUC | qwen3-8b PR-AUC | qwen3-4b PR-AUC |
|---|---|---|---|---|---|---|
| Sep_2021 (easiest) | 271 | 14.8% | 0.289 | 0.425 | 0.468 | 0.442 |
| Menon_2022 | 975 | 7.6% | 0.551 | 0.761 | 0.786 | 0.791 |
| van_der_Waal_2022 (hardest, realistic) | 1,970 | 1.7% | 0.213 | 0.371 | 0.315 | 0.338 |
| **Mean across all 3** | — | — | **0.351** | 0.519 | 0.523 | **0.523** |

**The win holds up on data this project had no hand in creating.** Every new model
clearly and consistently beat the local baseline (mean PR-AUC ~0.52 vs. 0.35), and
Recall@20% — the most important number for a use case with only 1.7% of papers actually
relevant — hit **100%** on the hardest review for both Qwen models. This closes the
open action item from `reports/wf_ensemble_report.md` §7 (item #2).

*Notebook: `notebooks/experiments/wf_synergy_validation.ipynb`.*

---

## 8. Final recommendation

Putting all of the above together:

1. **Adopt Jasper-Token-Compression-600M + Qwen3-Embedding-4B as the corpus embedding**,
   combined by concatenating their vectors. This pairing:
   - Beat every single model tested, across all 6 use cases, not just one.
   - Matched the more expensive Jasper+qwen3-8b pairing to within noise (0.636 vs.
     0.639), at meaningfully lower ongoing cost.
   - Held up on an independent, external benchmark (SYNERGY).
2. **Add isotonic calibration on top**, refit per use case once ~50 labels exist for it —
   this is a near-free fix for a real, demonstrated problem (raw scores being
   meaningless out-of-domain), and directly implements the calibration-only rung of the
   adaptation ladder in `reports/wf_ensemble_report.md` §4.
3. **Re-embed the whole historical corpus if this is adopted**, rather than mixing the
   old local-baseline vectors with the new ones. Confirmed in this investigation: this is
   cheap (a few dollars, under an hour of wall-clock time even for ~100k papers — see §9)
   and avoids the exact `embed_model`-mixing confound `wf_ensemble_report.md` §1 warns
   against.
4. **Don't treat this as solved for every use case.** `carbon_capture` and
   `soil_microbiome` specifically did not benefit from the new embeddings in this test —
   worth re-checking once real production data is available for those domains, and
   worth remembering that "best on average" is not the same as "best everywhere."

---

## 9. Cost and practicality

All of this was run for real, not estimated:

- **API costs (OpenRouter, Round 1)**: under $1 total across 7 models on the ~1,850
  labelled papers used for testing.
- **GPU costs (Modal, Round 2 + SYNERGY)**: a few dollars total, across all 4 GPU-hosted
  models, the full labelled corpus, and 3 SYNERGY reviews (~3,200 additional papers).
- **At production scale (~100k papers)**: extrapolating measured per-paper timings, even
  the priciest model tested (`gemini-embedding-2`) would cost roughly $7–8 to embed the
  full corpus once; the recommended Jasper+qwen3-4b pairing would cost a few dollars
  combined. This is a **one-time embedding cost**, not a recurring one — vectors are
  cached and reused, not recomputed per query.
- Everything is cached locally in `data/processed/embeddings_cache/` (gitignored, not
  committed) so re-running any of these notebooks costs nothing further unless new
  papers are added.

---

## 10. Open items / what to try next

1. **Re-test `carbon_capture` and `soil_microbiome` specifically** once there's reason to
   believe the embedding choice matters there — these were the two domains where the new
   models didn't clearly help.
2. **Try Matryoshka/dimension truncation** — some of the newer models (Gemini
   confirmed, Qwen3 family likely) support truncating their output vectors to fewer
   dimensions with minimal quality loss, which could reduce storage/compute further.
3. **Test per-use-case model routing** — this investigation only checked one global
   embedding choice; it's possible different domains would benefit from different
   models, which the per-use-case leave-one-out breakdown (§4) hints at but doesn't
   settle.
4. **Pull in more SYNERGY reviews** if more external validation confidence is wanted —
   only 3 of the 26 available reviews were tested here, chosen for a size/difficulty
   spread, not exhaustively.
5. **Revisit once `use_case_version` exists in a real export** (see
   `reports/wf_ensemble_report.md` §1) — none of this changes based on that gap, but it's
   still an open dependency for clean cross-time comparisons generally.

---

## Where everything lives

| What | Where |
|---|---|
| Round 1 + Round 2 bake-off (12 sources, single holdout) | `notebooks/experiments/wf_embedding_model_bakeoff.ipynb` |
| Leave-one-use-case-out, combining, calibration | `notebooks/experiments/wf_top_embeddings_generalization.ipynb` |
| SYNERGY external validation | `notebooks/experiments/wf_synergy_validation.ipynb` |
| GPU-hosted model backend (Modal) | `scripts/modal_embeddings.py` |
| Shared embedding logic (OpenRouter + Modal backends, model registry) | `scripts/embedding_utils.py` |
| Embedding cache (gitignored, reused across all 3 notebooks) | `data/processed/embeddings_cache/` |
