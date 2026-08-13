"""Population metrics from a case-control sample.

`benchset_loader.case_control_sample` scores every positive and a random `1/w` share of the
negatives. Recall and ROC-AUC survive that untouched; everything that counts negatives above
the line does not, and has to be reweighted. These are the reweighted versions, written to
match the repo's existing definitions exactly rather than invent new ones:

  - F2 is `5PR / (4P + R)`, the same as `sklearn.fbeta_score(beta=2)`.
  - WSS@95 is `(N - screened)/N - (1 - 0.95)` — Cohen et al. 2006, copied line for line from
    `embedding_utils.retrieval_ranking_metrics` so the numbers here are comparable to every
    WSS in the repo. Only `N` and `screened` become weight sums.

Ties are the thing to be careful about
--------------------------------------
Verbalised LLM scores are extremely lumpy — the pilot measured a **tie fraction above 0.99**,
9-16 distinct values across 1,848 rows. A rank metric computed by sorting such a score puts
the WSS cutoff *inside* a tie block thousands of rows wide, and the answer then depends on
whatever order the sort happened to produce. That is not a measurement.

So `wss_at` resolves ties by expectation: within a block of equal scores it assumes the
positives are uniformly distributed and interpolates. `tie_frac` is returned next to every
ranking number so a reader can see how much of it is interpolation. `work_at_own` needs
none of this and is the honest deployment number: the model's own verdict is one operating
point, and (recall, fraction screened) describes it completely.
"""

from __future__ import annotations

import numpy as np

BETA2 = 2.0


def weighted_prf(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict:
    """Precision / recall / F2 at population prevalence, from the reweighted confusion."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    w = np.asarray(w, dtype=float)
    tp = w[y & pred].sum()
    fp = w[~y & pred].sum()
    fn = w[y & ~pred].sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denom = 4 * precision + recall
    f2 = (5 * precision * recall / denom) if denom else 0.0
    return {
        "precision": float(precision), "recall": float(recall), "f2": float(f2),
        "pos_rate": float(w[pred].sum() / w.sum()),
    }


def f2_at(y, score, w, t: float) -> float:
    return weighted_prf(y, np.asarray(score) >= t, w)["f2"]


def f2_optimal(y, score, w, thresholds: np.ndarray | None = None) -> tuple[float, float]:
    """Best weighted F2 over a threshold sweep, and the threshold that got it.

    This is an **oracle** — it keeps the best threshold on the same rows it scored, exactly
    like `analyze_ensemble_v2_experiments.f2_optimal` whose 0.893 the pilot compared against.
    Report it for both sides or neither, and never without the honest operating point beside it.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    best = max(thresholds, key=lambda t: f2_at(y, score, w, t))
    return float(f2_at(y, score, w, best)), float(best)


def all_positive_f2(y: np.ndarray, w: np.ndarray) -> float:
    """F2 of marking every paper relevant: 5p/(4p+1) at population prevalence p.

    The number every other F2 on this surface has to be read against. On TIRI's 57.7% pools
    it is 0.872 and swamps everything; at set A's 2.19% it is 0.101 and F2 finally
    discriminates. That difference is the entire reason for running here.
    """
    y = np.asarray(y).astype(bool)
    w = np.asarray(w, dtype=float)
    p = w[y].sum() / w.sum()
    return float(5 * p / (4 * p + 1))


def tie_frac(score: np.ndarray) -> float:
    _, counts = np.unique(np.asarray(score), return_counts=True)
    return float(counts[counts > 1].sum() / len(score))


def _blocks(score: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Descending distinct-score blocks: (positives in block, weight in block)."""
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    s, yy, ww = np.asarray(score)[order], np.asarray(y)[order].astype(float), np.asarray(w)[order]
    edges = np.flatnonzero(np.diff(s)) + 1
    return zip(np.split(yy, edges), np.split(ww, edges))


def wss_at(y, score, w, target_recall: float = 0.95) -> dict:
    """Work Saved over Sampling, reweighted to the population and tie-aware.

    `screened` is cumulative *weight* down the ranking — "how many papers a human would
    actually have to read" — not row count. Getting that wrong on a case-control sample
    inflates WSS enormously, because the sampled negatives stand for 25x their number on
    `brouwer_2019`.
    """
    y = np.asarray(y).astype(float)
    w = np.asarray(w, dtype=float)
    n_pos, n_w = y.sum(), w.sum()
    if n_pos == 0 or n_pos == len(y):
        return {"wss": float("nan"), "screened_frac": float("nan"), "tie_frac": tie_frac(score)}

    target = np.ceil(target_recall * n_pos)
    seen_pos = seen_w = 0.0
    screened = n_w
    for by, bw in _blocks(score, y, w):
        p, wt = by.sum(), bw.sum()
        if seen_pos + p >= target:
            # Cutoff lands inside this block. Under random ordering within a block of equal
            # scores the positives are uniformly spread, so the expected weight read to
            # reach the target is linear in the share of the block's positives needed.
            share = (target - seen_pos) / p if p else 1.0
            screened = seen_w + wt * share
            break
        seen_pos += p
        seen_w += wt
    return {
        "wss": float((n_w - screened) / n_w - (1 - target_recall)),
        "screened_frac": float(screened / n_w),
        "tie_frac": tie_frac(score),
    }


def recall_at(y, score, w, frac: float) -> float:
    """Recall after reading the top `frac` of the corpus *by weight*, tie-interpolated."""
    y = np.asarray(y).astype(float)
    w = np.asarray(w, dtype=float)
    n_pos, budget = y.sum(), frac * w.sum()
    if n_pos == 0:
        return float("nan")
    seen_pos = seen_w = 0.0
    for by, bw in _blocks(score, y, w):
        p, wt = by.sum(), bw.sum()
        if seen_w + wt >= budget:
            seen_pos += p * ((budget - seen_w) / wt if wt else 0.0)
            break
        seen_pos += p
        seen_w += wt
    return float(min(seen_pos / n_pos, 1.0))


def work_at_own(y, pred, w) -> dict:
    """The model's own verdict as a (recall, work) pair. No threshold, no ties, no oracle.

    A screening tool's real promise is "read this pile instead of that one". At 2.19%
    prevalence that is what a reviewer buys, and unlike WSS@95 it does not require squeezing
    a rank out of a 10-valued score.
    """
    r = weighted_prf(y, pred, w)
    return {"recall": r["recall"], "screened_frac": r["pos_rate"],
            "work_saved": 1.0 - r["pos_rate"], "precision": r["precision"], "f2": r["f2"]}
