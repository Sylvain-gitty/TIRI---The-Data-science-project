"""Three whitepaper figures, each answering one question a reviewer asked of the report.

Run from the repo root with the project venv active:

    python scripts/make_whitepaper_figures.py

WHY THESE THREE
---------------
The whitepaper carried three claims that a reader could not check from the prose:

1. "Ten features beat the 384-number embedding" — the central mechanism, stated as a
   number with no picture of what the two feature sets actually read.
2. "Real Use Cases beat wrong Use Cases on 5 of 6" — the one failure was not named.
3. "An estimated practical ceiling sits around 86%" — an unsourced figure. What *is*
   measurable is the opposite bound: the score a model gets for doing nothing at all.

Every number below is copied from an executed report, not recomputed here, except the
trivial-policy F2 in figure 3, which is computed from `papers_fe_slim.parquet` on the
same split notebook 06 used (StratifiedGroupKFold(5), grouped by first author,
stratified on use case + label, fold 0) so it sits on exactly the same rows as the
model scores it is drawn against.

SOURCES
-------
figs 1-2  reports/wf_tier1b_lexical_control.md  §1  (leave-one-Use-Case-out,
          plain LogisticRegression, 6 rotations)
fig 3     notebooks/main/06_baseline_logreg.ipynb §6  (baseline F2)
          notebooks/main/08_ensemble_pooled.ipynb §7  (ensemble F2)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedGroupKFold

REPORTS = Path("reports")
DATA = Path("data/processed/papers_fe_slim.parquet")

GREEN, RED, GREY, BLUE = "#2e9e33", "#d62728", "#b3b3b3", "#3f6fb5"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 12,
    }
)

# --- Leave-one-Use-Case-out ROC-AUC, from wf_tier1b_lexical_control.md §1 -------------
USE_CASE_LABELS = {
    "cement_binders": "Low-carbon cement",
    "carbon_capture": "Carbon capture",
    "ner": "Named entity recognition",
    "soil_microbiome": "Soil microbiome",
    "tech_forecasting": "Technology prediction",
    "solar_leo": "Solar cells for satellites",
}
REAL = {
    "cement_binders": 0.890,
    "carbon_capture": 0.744,
    "ner": 0.673,
    "soil_microbiome": 0.594,
    "tech_forecasting": 0.540,
    "solar_leo": 0.413,
}
WRONG = {
    "cement_binders": 0.521,
    "carbon_capture": 0.473,
    "ner": 0.483,
    "soil_microbiome": 0.490,
    "tech_forecasting": 0.504,
    "solar_leo": 0.455,
}
EMBEDDING_MEAN, REAL_MEAN, WRONG_MEAN = 0.537, 0.642, 0.488


def figure_1_ten_beat_384() -> Path:
    """The central mechanism: which *kind* of feature survives a new Use Case."""
    rows = [
        ("Ten features that compare\nthe paper to the Use Case", REAL_MEAN, GREEN),
        ("The full 384-number embedding\nof the paper on its own", EMBEDDING_MEAN, BLUE),
        ("The same ten features, scored against\na deliberately wrong Use Case", WRONG_MEAN, RED),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ypos = np.arange(len(rows))[::-1]
    ax.barh(ypos, [r[1] for r in rows], color=[r[2] for r in rows], height=0.55)
    for y, (_, val, _) in zip(ypos, rows):
        ax.text(val + 0.006, y, f"{val:.3f}", va="center", fontweight="bold")

    ax.axvline(0.5, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(0.5, len(rows) - 0.42, "0.500 — a coin flip", color="#444444", fontsize=9.5,
            ha="center", va="bottom")
    ax.set_ylim(-0.55, len(rows) - 0.25)

    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlim(0.40, 0.70)
    ax.set_xlabel("How often the ranking puts an interesting paper above a boring one,\n"
                  "on a Use Case the model has never seen (ROC-AUC)")
    ax.set_title("Ten beat 384: it was never the number of features, it was the kind\n"
                 "Averaged over six rotations — each Use Case held out in turn",
                 loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    out = REPORTS / "wf_whitepaper_1_ten_beat_384.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_2_wrong_use_case_control() -> Path:
    """The adversarial control, per Use Case, with the one failure named."""
    order = sorted(REAL, key=lambda k: REAL[k] - WRONG[k], reverse=True)
    x = np.arange(len(order))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.bar(x - w / 2, [REAL[k] for k in order], w, label="Scored against its real Use Case",
           color=GREEN)
    ax.bar(x + w / 2, [WRONG[k] for k in order], w,
           label="Scored against a deliberately wrong Use Case", color=RED)

    ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(5.62, 0.5, "0.500\na coin flip", color="#444444", fontsize=9.5, ha="left",
            va="center")
    ax.set_xlim(-0.7, 6.35)

    for i, k in enumerate(order):
        gap = REAL[k] - WRONG[k]
        top = max(REAL[k], WRONG[k])
        ax.text(i, top + 0.014, f"{gap:+.3f}", ha="center", fontsize=9.5,
                fontweight="bold", color=GREEN if gap > 0 else RED)

    fail = order.index("solar_leo")
    ax.annotate(
        "The one failure. Both arms land below a coin flip:\n"
        "there was nothing in the text for either Use Case to read",
        xy=(fail - 0.19, 0.425), xytext=(2.55, 0.755),
        fontsize=9.5, color=RED, ha="left",
        arrowprops={"arrowstyle": "->", "color": RED, "linewidth": 1.2},
    )

    ax.set_xticks(x)
    ax.set_xticklabels([USE_CASE_LABELS[k].replace(" ", "\n", 1) for k in order], fontsize=9.5)
    ax.set_ylim(0.30, 0.97)
    ax.set_ylabel("ROC-AUC on the held-out Use Case")
    ax.set_title("The wrong-Use-Case control: real beats wrong on 5 of 6\n"
                 "If the features were secretly measuring something generic, the red bars\n"
                 "would match the green ones. Mean gap +0.155",
                 loc="left")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    out = REPORTS / "wf_whitepaper_2_wrong_use_case_control.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def trivial_policy_f2() -> tuple[float, float, int]:
    """F2 for 'call every paper interesting', on notebook 06's own holdout rows."""
    df = pd.read_parquet(DATA)
    strat = df["use_case_key"].astype(str) + "__" + df["y"].astype(str)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    _, holdout_idx = next(iter(cv.split(df, strat, groups=df["first_author"])))
    y = df.iloc[holdout_idx]["y"].astype(int).to_numpy()
    return (
        float(fbeta_score(y, np.ones_like(y), beta=2, zero_division=0)),
        float(y.mean()),
        len(y),
    )


def figure_3_floor_not_ceiling(trivial: float, prevalence: float, n: int) -> Path:
    """What a score has to beat here — the do-nothing policy, not an invented ceiling."""
    stages = ["Train\n(60%)", "Validate\n(20%)", "Test\n(20%)"]
    baseline = [0.841, 0.820, 0.790]
    ensemble = [0.958, 0.906, 0.892]

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    x = np.arange(3)
    ax.plot(x, ensemble, marker="o", linewidth=2, color=BLUE,
            label="Ensemble (CatBoost + random forest + logistic regression)")
    ax.plot(x, baseline, marker="s", linewidth=2, color=GREY,
            label="Baseline (logistic regression, untuned)")

    ax.axhline(trivial, color=RED, linestyle="--", linewidth=1.6)
    ax.text(
        -0.18, trivial + 0.008,
        f'"Call every paper interesting" — F2 = {trivial:.3f}',
        color=RED, fontsize=9.5, va="bottom",
    )

    for xi, (b, e) in enumerate(zip(baseline, ensemble)):
        ax.text(xi, e + 0.011, f"{e:.3f}", ha="center", fontsize=9.5, color=BLUE,
                fontweight="bold")
        ax.text(xi, b - 0.022, f"{b:.3f}", ha="center", fontsize=9.5, color="#666666",
                fontweight="bold")

    ax.annotate("", xy=(2.3, ensemble[-1]), xytext=(2.3, trivial),
                arrowprops={"arrowstyle": "<->", "color": BLUE, "linewidth": 1.2})
    ax.text(2.38, 0.938, f"{ensemble[-1] - trivial:+.3f} — the ensemble's\nentire margin over "
            "doing nothing", fontsize=9, color=BLUE, va="center")
    ax.annotate("", xy=(2.3, baseline[-1]), xytext=(2.3, trivial),
                arrowprops={"arrowstyle": "<->", "color": "#777777", "linewidth": 1.2})
    ax.text(2.38, 0.795, f"{baseline[-1] - trivial:+.3f} — the baseline\nnever reaches the floor",
            fontsize=9, color="#666666", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_xlim(-0.25, 3.35)
    ax.set_ylim(0.655, 1.0)
    ax.set_ylabel("F2 (recall weighted twice precision)")
    ax.set_title(
        "The floor that can be measured, in place of a ceiling that cannot\n"
        f"The {n} unseen test papers are {prevalence:.0%} interesting, so marking every one of them\n"
        f"interesting already scores F2 = {trivial:.3f}. That is the number a model has to beat",
        loc="left",
    )
    ax.legend(loc="lower left", fontsize=9.5, bbox_to_anchor=(0.0, 0.02))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    out = REPORTS / "wf_whitepaper_3_f2_floor.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    trivial, prevalence, n = trivial_policy_f2()
    print(f"Trivial 'everything is interesting' policy on notebook 06's holdout: "
          f"n={n}, {prevalence:.1%} interesting, F2={trivial:.3f}")
    for out in (
        figure_1_ten_beat_384(),
        figure_2_wrong_use_case_control(),
        figure_3_floor_not_ceiling(trivial, prevalence, n),
    ):
        print("wrote", out)


if __name__ == "__main__":
    main()
