# Multi-seed verification of Ensemble v2 experiments

> **Status: current, supporting.** The multi-seed, noise-floor-gated check on [`wf_ensemble_v2_experiments.md`](wf_ensemble_v2_experiments.md)'s single-seed findings. **Read this before acting on any number in that file.**

Same three comparisons as `reports/wf_ensemble_v2_experiments.md`, repeated across 5/5 seeds instead of one, applying this project's own noise-floor discipline (`CONTEXT.md` §5: treat any gap below ~0.03 ROC-AUC as not established).

## Experiment A — tuned vs. screening CatBoost, across seeds

| use_case                   | mean   | std   |
|----------------------------|--------|-------|
| Carbon Capture             | 0.011  | 0.004 |
| Low-Carbon Cement          | 0.007  | 0.008 |
| Named Entity Recognition   | 0.025  | 0.015 |
| Soil Microbiome            | 0.024  | 0.007 |
| Solar Cells for Satellites | 0.024  | 0.011 |
| Technology Prediction      | -0.001 | 0.021 |

Overall: mean delta **+0.0153** (sd across all seed x use_case rows: 0.0150). Clears the noise floor: **no**.

## Experiment B — weighted combiner vs. fixed 50/50, across seeds

| use_case                   | mean  | std   |
|----------------------------|-------|-------|
| Carbon Capture             | 0.002 | 0.002 |
| Low-Carbon Cement          | 0.007 | 0.005 |
| Named Entity Recognition   | 0.006 | 0.002 |
| Soil Microbiome            | 0.014 | 0.003 |
| Solar Cells for Satellites | 0.011 | 0.006 |
| Technology Prediction      | 0.006 | 0.004 |

Overall: mean delta **+0.0078** (sd: 0.0053). Clears the noise floor: **no**. Still an in-sample weight choice (no inner/outer split) — an upper bound even at 5 seeds.

Best weight per use case, mean +/- sd across seeds (0 = pure LogReg, 1 = pure CatBoost):

| use_case                   | mean  | std   |
|----------------------------|-------|-------|
| Carbon Capture             | 0.560 | 0.330 |
| Low-Carbon Cement          | 0.900 | 0.120 |
| Named Entity Recognition   | 0.670 | 0.380 |
| Soil Microbiome            | 0.950 | 0.040 |
| Solar Cells for Satellites | 0.830 | 0.100 |
| Technology Prediction      | 0.380 | 0.500 |

## Experiment C — Platt calibration's effect on F2@0.5, across seeds

| use_case                   | mean   | std   |
|----------------------------|--------|-------|
| Carbon Capture             | 0.001  | 0.002 |
| Low-Carbon Cement          | -0.000 | 0.001 |
| Named Entity Recognition   | 0.023  | 0.004 |
| Soil Microbiome            | -0.050 | 0.040 |
| Solar Cells for Satellites | 0.026  | 0.005 |
| Technology Prediction      | 0.005  | 0.006 |

Overall: mean F2@0.5 lift from calibration **+0.0006** (sd: 0.0295); mean remaining gap to F2@t* after calibration: **+0.0780**.

## What this actually shows — read per-use-case, not just the pooled mean

The blanket "does the pooled mean clear 0.03" check hides a more useful pattern: for several
use cases, the effect size is small in absolute terms but tight relative to *its own* seed
noise (mean well above its own sd) — a real, reproducible per-use-case effect, even where the
pooled-across-everything mean doesn't clear the generic bar. Reading each experiment that way:

**Experiment A (tuned CatBoost) is real on 3-4/6 use cases, not on all 6.** Soil Microbiome
(+0.024, sd 0.007 — ratio ~3.4), Solar Cells for Satellites (+0.024, sd 0.011 — ratio ~2.2),
and Carbon Capture (+0.011, sd 0.004 — ratio ~2.75) all show a tight, likely-real gain.
Named Entity Recognition (+0.025, sd 0.015) is directionally there but noisier. Low-Carbon
Cement (+0.007, sd 0.008) and Technology Prediction (−0.001, sd 0.021) show no reliable
effect — un-cheaping the hyperparameters doesn't help those two, and it may make Technology
Prediction very slightly worse (within its own noise, not a real loss either).

**Experiment B's most useful finding survived the multi-seed check, and it's not "0.009 mean
gain" — it's which use cases have a *stable* optimal weight.** Soil Microbiome (mean weight
0.95, sd 0.04), Low-Carbon Cement (0.90, sd 0.12), and Solar Cells for Satellites (0.83, sd
0.10) all consistently favor CatBoost heavily over LogReg, seed after seed — a low-variance,
reproducible pattern, not noise. Carbon Capture (0.56, sd 0.33), Named Entity Recognition
(0.67, sd 0.38), and Technology Prediction (0.38, sd 0.50) swing across nearly the entire
0-1 range from seed to seed — there is no stable "best weight" for these three, and forcing
one would be fitting noise. **The actionable version of Experiment B is not "always use the
best weight" — it's "weight toward CatBoost specifically for the three use cases with a
low-variance preference, keep 50/50 for the other three."**

**Experiment C's single-seed finding did not survive.** The single-seed run showed a
consistent small positive lift from calibration (+0.010 mean). At 5 seeds the true mean is
statistically indistinguishable from zero (+0.0006, sd 0.0295 — the noise is 50x the mean).
Worse: Soil Microbiome shows a *negative* mean (−0.050, sd 0.040) — in-sample Platt scaling
can make F2@0.5 worse, not better, on the use case that's already hardest to separate
topically (`reports/wf_featureengineering_review.md` §5). Named Entity Recognition (+0.023,
sd 0.004) and Solar Cells for Satellites (+0.026, sd 0.005) are the two use cases where
calibration shows a tight, likely-real small benefit. **This is exactly the scenario the
whole exercise was built to catch** — a plausible-looking single-seed result that mostly
evaporates under repetition.

## Bottom line for what changes in v2

1. **Ship the tuned CatBoost setting (iterations=150, depth=4)** — real gain on at least 3
   use cases, harmless elsewhere, no reason to keep the iterations=50 screening default now
   that only one feature set needs fitting.
2. **Use a per-use-case combiner weight only for Soil Microbiome, Low-Carbon Cement, and
   Solar Cells for Satellites** (favor CatBoost, ~0.85-0.95); keep 50/50 for Carbon Capture,
   Named Entity Recognition, and Technology Prediction, where "best weight" isn't a stable
   quantity. Still needs a properly nested (not in-sample) weight estimate before shipping,
   per the same caveat as the single-seed round.
3. **Don't ship blanket calibration as a fix for the naive-threshold problem.** It's a
   near-zero effect on average and a measured negative on Soil Microbiome specifically. If
   Named Entity Recognition or Solar Cells for Satellites need a usable 0.5-ish cutoff, a
   *per-use-case* calibration decision is defensible; a repo-wide one is not.
