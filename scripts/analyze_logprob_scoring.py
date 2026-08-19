"""Does token-logprob scoring rank better than a verbalised 0-100 score?

The pilot lost exactly one comparison — ranking, 0.821 vs the ensemble's 0.865 — and there
was reason to think the loss was partly an artifact of how the score was *elicited* rather
than of how well the model judged. Asked for a number, every model answered in round
figures: 9-16 distinct values across 1,848 rows, tie fraction above 0.99. ROC-AUC gives
tied pairs half credit, so a lumpy score is penalised on ties alone.

P2lp asks the same question with the same F2-asymmetry framing but elicits a single token,
and the decision is read from the token distribution instead of from a number the model
writes down.

**Log-odds, not the renormalised probability.** At temperature 0 these models are extremely
peaked (measured: NO at -0.0 with the runner-up at -14.5), so P(YES) saturates to a near
binary 0/1 and discards the granularity this variant exists to recover — on the smoke
subset only 3 of 36 rows landed strictly inside (0.01, 0.99). The unsquashed margin
logprob(YES) - logprob(NO) keeps it: a row where YES wins by 2 nats and one where it wins
by 25 are genuinely different confidences. Ranking metrics only need an ordering.

Provider caveat, stated because it is not fully controlled
----------------------------------------------------------
Only `gpt-oss-20b` has both variants on the same provider (CoreWeave). gemma moved
Friendli->CoreWeave and qwen Alibaba->Parasail, because neither original pin returns
logprobs at all. So for two of three models the comparison crosses providers. If all three
move in the same direction, the provider change is not what is driving it — that is the
check, and it is why gpt-oss is worth keeping in the comparison despite being the weakest
model in the pilot.

Usage:
    python scripts/analyze_logprob_scoring.py
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
OUT_MD = REPO / "reports" / "wf_llm_logprob_scoring.md"
ENSEMBLE_AUC = 0.865
MODELS = ["openai/gpt-oss-20b", "google/gemma-4-31b-it", "qwen/qwen3.5-397b-a17b"]
SAME_PROVIDER = {"openai/gpt-oss-20b"}  # the only clean within-provider pair


def rank_stats(y, s):
    _, counts = np.unique(s, return_counts=True)
    return {
        "roc_auc": roc_auc_score(y, s),
        "n_distinct": int(len(np.unique(s))),
        "tie_frac": float(counts[counts > 1].sum() / len(s)),
    }


def main() -> None:
    grid = pd.read_parquet(GRID)
    if "score_logodds" not in grid.columns:
        raise SystemExit("no score_logodds column - run the P2lp cells first")

    lines: list[str] = []

    def emit(t: str = "") -> None:
        print(t)
        lines.append(t)

    emit("# Token-logprob scoring vs a verbalised 0-100 score")
    emit()
    emit("Ranking signal for P2lp is `logprob(YES) - logprob(NO)`, not the renormalised "
         "probability, which saturates at temperature 0. Both variants use the same "
         "F2-asymmetry framing; only the elicitation differs.")
    emit()

    rows = []
    for model in MODELS:
        for variant, col in (("P2", "score"), ("P2lp", "score_logodds")):
            g = grid[(grid.model == model) & (grid.variant == variant)]
            g = g[g[col].notna()]
            if g.empty:
                continue
            per_uc = []
            for uc, sub in g.groupby("use_case_key"):
                y = sub["y"].to_numpy().astype(int)
                if len(np.unique(y)) < 2:
                    continue
                per_uc.append({"use_case": uc, **rank_stats(y, sub[col].to_numpy())})
            d = pd.DataFrame(per_uc)
            pred = g["pred"].astype(float).to_numpy()
            m = ~np.isnan(pred)
            rows.append({
                "model": model.split("/")[-1],
                "variant": variant,
                "mean_auc": d.roc_auc.mean(),
                "n_distinct": d.n_distinct.mean(),
                "tie_frac": d.tie_frac.mean(),
                "f2_own": fbeta_score(g["y"].to_numpy().astype(int)[m],
                                      pred[m].astype(int), beta=2, zero_division=0),
                "coverage": round(len(g) / 1848, 3),
                "same_provider": model in SAME_PROVIDER,
            })

    res = pd.DataFrame(rows)
    emit("## Per model")
    emit()
    emit(to_md(res.round(3), ""))
    emit()

    piv = res.pivot_table(index="model", columns="variant",
                          values=["mean_auc", "n_distinct", "tie_frac"])
    emit("## The comparison")
    emit()
    comp = pd.DataFrame({
        "auc_verbalised": piv[("mean_auc", "P2")],
        "auc_logprob": piv[("mean_auc", "P2lp")],
        "auc_gain": piv[("mean_auc", "P2lp")] - piv[("mean_auc", "P2")],
        "distinct_verbalised": piv[("n_distinct", "P2")],
        "distinct_logprob": piv[("n_distinct", "P2lp")],
        "ties_verbalised": piv[("tie_frac", "P2")],
        "ties_logprob": piv[("tie_frac", "P2lp")],
    })
    emit(to_md(comp.round(3), "model"))
    emit()

    gain = comp.auc_gain
    agree = "all three" if (gain > 0).all() else f"{int((gain > 0).sum())} of {len(gain)}"
    emit(f"Mean AUC change **{gain.mean():+.3f}**, positive on **{agree}** models. "
         f"Best logprob AUC **{comp.auc_logprob.max():.3f}** vs the ensemble's "
         f"**{ENSEMBLE_AUC:.3f}** (gap {comp.auc_logprob.max() - ENSEMBLE_AUC:+.3f}).")
    emit()
    clean = comp.loc[[m.split("/")[-1] for m in SAME_PROVIDER]]
    emit(f"On the one within-provider pair (`gpt-oss-20b`, CoreWeave both sides) the change "
         f"is **{clean.auc_gain.iloc[0]:+.3f}** — the control for the provider switch the "
         "other two models required.")
    emit()

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
