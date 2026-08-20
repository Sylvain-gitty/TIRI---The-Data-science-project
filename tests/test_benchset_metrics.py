"""Invariants of the population-metric layer used for the external benchmark.

This module is where a case-control sample gets reweighted to real prevalence, so its bugs
are the expensive kind: they do not crash, they just report a better number than reality.
The tests below pin the arithmetic, the weighting, and the degenerate-input behaviour.

Two of them lock figures that the README and CONTEXT.md quote in prose.
"""

from __future__ import annotations

import numpy as np
import pytest

import benchset_metrics as bm


# --------------------------------------------------------------------------- weighted_prf


def test_weighted_prf_on_a_perfect_prediction():
    y = np.array([1, 1, 0, 0])
    w = np.ones(4)
    r = bm.weighted_prf(y, y, w)
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(1.0)
    assert r["f2"] == pytest.approx(1.0)


def test_weighted_prf_f2_weights_recall_four_times_precision():
    """F2 = 5PR / (4P + R). Half the positives found, no false positives."""
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 0, 0])
    r = bm.weighted_prf(y, pred, np.ones(4))
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(0.5)
    assert r["f2"] == pytest.approx(5 * 1.0 * 0.5 / (4 * 1.0 + 0.5))   # 0.5555...


def test_weights_change_precision_on_identical_rows():
    """The whole point of the layer: a sampled negative stands for many real ones.

    Same rows, same predictions — only the negative's weight changes. Precision must fall,
    because in the population that false positive is 25 papers a human reads for nothing.
    If this ever returns the same number twice, the reweighting is not being applied and
    every external metric in the repo is optimistic.
    """
    y = np.array([1, 0])
    pred = np.array([1, 1])
    unweighted = bm.weighted_prf(y, pred, np.array([1.0, 1.0]))["precision"]
    reweighted = bm.weighted_prf(y, pred, np.array([1.0, 25.0]))["precision"]
    assert unweighted == pytest.approx(0.5)
    assert reweighted == pytest.approx(1 / 26)
    assert reweighted < unweighted


def test_weighted_prf_returns_zero_not_nan_when_nothing_is_predicted_positive():
    """A model that flags nothing has precision 0, not a NaN that poisons a mean."""
    r = bm.weighted_prf(np.array([1, 0]), np.array([0, 0]), np.ones(2))
    assert r["precision"] == 0.0
    assert r["recall"] == 0.0
    assert r["f2"] == 0.0
    assert not np.isnan(r["f2"])


# --------------------------------------------------------------------------- the F2 floor


def test_all_positive_f2_is_the_closed_form():
    """F2 of marking everything relevant is 5p/(4p+1) at population prevalence p."""
    for p in (0.01, 0.0219, 0.1, 0.5, 0.577, 0.9):
        y = np.array([1, 0])
        w = np.array([p, 1 - p])
        assert bm.all_positive_f2(y, w) == pytest.approx(5 * p / (4 * p + 1))


def test_all_positive_f2_reproduces_the_two_figures_the_repo_quotes():
    """Both numbers appear in prose in README and CONTEXT.md 3.

    They carry the argument for why the external benchmark exists at all: at TIRI's own
    pool prevalence "mark everything relevant" already scores 0.872, so F2 cannot tell
    anything apart; at the benchmark's 2.19% it scores 0.101 and F2 finally discriminates.
    """
    def floor(p):
        return bm.all_positive_f2(np.array([1, 0]), np.array([p, 1 - p]))

    assert floor(0.577) == pytest.approx(0.872, abs=0.001)     # TIRI's own pools
    assert floor(0.0219) == pytest.approx(0.101, abs=0.001)    # benchset_v1 set A


# --------------------------------------------------------------------------- oracle sweep


def test_f2_optimal_is_an_upper_bound_on_any_fixed_threshold():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.2).astype(int)
    score = rng.random(200)
    w = rng.uniform(1, 30, 200)

    best, thr = bm.f2_optimal(y, score, w)
    assert 0.0 <= thr <= 1.0
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert best >= bm.f2_at(y, score, w, t) - 1e-12


# --------------------------------------------------------------------------- ties


def test_tie_frac_at_both_extremes():
    assert bm.tie_frac(np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(0.0)
    assert bm.tie_frac(np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(1.0)
    assert bm.tie_frac(np.array([0.1, 0.1, 0.2, 0.3])) == pytest.approx(0.5)


# --------------------------------------------------------------------------- ranking metrics


def test_wss_is_high_for_a_perfect_ranking():
    y = np.array([1] * 5 + [0] * 95)
    w = np.ones(100)

    # 5 positives, target_recall 0.95 -> target = ceil(4.75) = 5, all of them in the top 5
    # rows. So screened_frac = 0.05 and wss = (1 - 0.05) - (1 - 0.95) = 0.90 exactly.
    perfect = np.concatenate([np.linspace(1.0, 0.9, 5), np.linspace(0.5, 0.0, 95)])
    out = bm.wss_at(y, perfect, w, 0.95)
    assert out["screened_frac"] == pytest.approx(0.05)
    assert out["wss"] == pytest.approx(0.90)


def test_wss_of_an_uninformative_ranking_is_zero():
    """WSS is defined against random sampling, so no ranking information must score 0.

    One flat score for every paper. Reaching 95% recall by reading a uniformly-mixed pile
    means reading 95% of it, so the work saved (0.05) exactly cancels the (1 - 0.95) term.
    """
    y = np.array([1] * 100 + [0] * 900)
    w = np.ones(1000)
    out = bm.wss_at(y, np.zeros(1000), w, 0.95)
    assert out["screened_frac"] == pytest.approx(0.95)
    assert out["wss"] == pytest.approx(0.0)
    assert out["tie_frac"] == pytest.approx(1.0)


def test_wss_tie_interpolation_is_conservative_when_the_target_rounds_to_every_positive():
    """A documented edge of the linear tie interpolation, pinned so it cannot drift silently.

    `target = ceil(target_recall * n_pos)`, so with few positives it can round up to *all*
    of them (5 positives -> ceil(4.75) = 5). The share of a tied block that must be read is
    then (target - seen) / p = 1.0, i.e. the whole block, giving WSS = -(1 - target_recall).

    A uniform-spread argument would say the fifth of five positives sits about 5/6 of the
    way down instead, so this is the pessimistic reading. That is the existing behaviour and
    published numbers rest on it; this test exists to document it, not to endorse it.
    """
    y = np.array([1] * 5 + [0] * 95)
    out = bm.wss_at(y, np.zeros(100), np.ones(100), 0.95)
    assert out["screened_frac"] == pytest.approx(1.0)
    assert out["wss"] == pytest.approx(-0.05)


def test_wss_returns_nan_on_a_degenerate_label_column():
    """No positives, or all positives, means WSS is undefined — not 0.0, not 1.0."""
    w = np.ones(4)
    assert np.isnan(bm.wss_at(np.zeros(4), np.array([1.0, 2, 3, 4]), w)["wss"])
    assert np.isnan(bm.wss_at(np.ones(4), np.array([1.0, 2, 3, 4]), w)["wss"])


def test_wss_screened_frac_counts_weight_not_rows():
    """The failure mode named in the docstring.

    A case-control sample's negatives stand for many times their number. If `screened`
    counted rows, WSS would be inflated enormously. Here every negative carries weight 50,
    so reading all 10 negatives is most of the population's reading effort even though it
    is only 10 rows.
    """
    y = np.array([0] * 10 + [1] * 2)
    score = np.concatenate([np.ones(10), np.zeros(2)])   # worst case: negatives ranked first
    w = np.array([50.0] * 10 + [1.0, 1.0])

    out = bm.wss_at(y, score, w, target_recall=0.95)
    # 500 of 502 total weight must be read before reaching the positives.
    assert out["screened_frac"] > 0.99
    assert out["wss"] < 0.0        # worse than random, and reported as such


def test_recall_at_full_budget_finds_everything():
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6])
    w = np.ones(4)
    assert bm.recall_at(y, score, w, 1.0) == pytest.approx(1.0)


def test_recall_at_is_monotonic_in_the_budget():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.15).astype(int)
    score = rng.random(300)
    w = rng.uniform(1, 40, 300)

    values = [bm.recall_at(y, score, w, f) for f in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0)]
    assert values == sorted(values)
    assert values[-1] == pytest.approx(1.0)


def test_work_at_own_recall_and_work_saved_are_consistent():
    y = np.array([1, 1, 0, 0, 0])
    pred = np.array([1, 0, 1, 0, 0])
    w = np.array([1.0, 1.0, 10.0, 10.0, 10.0])

    out = bm.work_at_own(y, pred, w)
    assert out["work_saved"] == pytest.approx(1.0 - out["screened_frac"])
    assert out["recall"] == pytest.approx(0.5)
