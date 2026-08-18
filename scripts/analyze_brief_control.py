"""Shuffled-brief falsification control: is the LLM reading the brief, or the paper?

The single most important check in the pilot. Every model in the grid was also run with
each use case's papers scored against a *different* use case's brief (a derangement — no
use case keeps its own). The logic is the one the rest of the repo already applies to
anything claiming to read the brief:

    A screener that genuinely conditions on the criteria should COLLAPSE when handed the
    wrong criteria. One that does not is ranking papers by generic quality, and every
    number measured on it means something other than what it says on the label.

This control is not a formality here — the same test passes decisively for the lexical
block on TIRI's six use cases (5/6, +0.155) and *fails* on SYNERGY, so it discriminates.

Reported per use case as well as in the mean, because a mean over six heterogeneous use
cases can be carried by one of them (CONTEXT.md §5).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_eval_utils import to_md  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GRID = REPO / "reports" / "wf_llm_grid_responses.parquet"
CONTROL = REPO / "reports" / "wf_llm_control_responses.parquet"
OUT_MD = REPO / "reports" / "wf_llm_brief_control.md"


def f2_optimal(y, s, thresholds=np.linspace(0.05, 0.95, 91)):
    return max(fbeta_score(y, (s >= t).astype(int), beta=2, zero_division=0)
               for t in thresholds)


def cell_metrics(g: pd.DataFrame) -> dict:
    ok = g[g["score"].notna()]
    y = ok["y"].to_numpy().astype(int)
    s = ok["score"].to_numpy()
    if len(np.unique(y)) < 2:
        return {}
    pred = ok["pred"].astype(float).to_numpy()
    m = ~np.isnan(pred)
    return {
        "roc_auc": roc_auc_score(y, s),
        "f2_at_t_star": f2_optimal(y, s),
        "f2_at_own": fbeta_score(y[m], pred[m].astype(int), beta=2, zero_division=0),
    }


def main() -> None:
    grid = pd.read_parquet(GRID)
    ctrl = pd.read_parquet(CONTROL)
    lines: list[str] = []

    def emit(t: str = "") -> None:
        print(t)
        lines.append(t)

    emit("# Shuffled-brief control — is the LLM reading the brief?")
    emit()
    emit("Each use case's papers scored against a **different** use case's brief "
         "(derangement, no fixed point). A brief-conditioned screener should collapse.")
    emit()

    summary = []
    for (model, variant), c in ctrl.groupby(["model", "variant"]):
        g = grid[(grid.model == model) & (grid.variant == variant)]
        rows = []
        for uc in sorted(c.use_case_key.unique()):
            real = cell_metrics(g[g.use_case_key == uc])
            fake = cell_metrics(c[c.use_case_key == uc])
            if not real or not fake:
                continue
            rows.append({
                "use_case": uc,
                "auc_real": real["roc_auc"], "auc_shuffled": fake["roc_auc"],
                "auc_drop": real["roc_auc"] - fake["roc_auc"],
                "f2own_real": real["f2_at_own"], "f2own_shuffled": fake["f2_at_own"],
                "f2own_drop": real["f2_at_own"] - fake["f2_at_own"],
            })
        d = pd.DataFrame(rows).set_index("use_case")
        emit(f"## {model} / {variant}")
        emit()
        emit(to_md(d.round(3), "use_case"))
        emit()
        n_drop = int((d.auc_drop > 0).sum())
        emit(f"Mean AUC {d.auc_real.mean():.3f} → **{d.auc_shuffled.mean():.3f}** "
             f"(drop **{d.auc_drop.mean():+.3f}**), degrades on **{n_drop}/{len(d)}** use cases. "
             f"Mean F2@own {d.f2own_real.mean():.3f} → **{d.f2own_shuffled.mean():.3f}** "
             f"(drop {d.f2own_drop.mean():+.3f}).")
        emit()
        # 0.5 is the reference that matters: a shuffled brief should leave a screener at
        # chance, not merely lower. Anything well above 0.5 is signal the brief did not carry.
        verdict = ("PASSES — collapses toward chance without its own brief"
                   if d.auc_shuffled.mean() < 0.60 and d.auc_drop.mean() > 0.15
                   else "PARTIAL — degrades, but retains signal the brief did not supply"
                   if d.auc_drop.mean() > 0.05
                   else "FAILS — the brief is not what it is reading")
        emit(f"**Verdict: {verdict}.**")
        emit()
        summary.append({"model": model, "variant": variant,
                        "auc_real": d.auc_real.mean(), "auc_shuffled": d.auc_shuffled.mean(),
                        "auc_drop": d.auc_drop.mean(), "degraded": f"{n_drop}/{len(d)}"})

    emit("## Summary")
    emit()
    emit(to_md(pd.DataFrame(summary).round(3), ""))
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
