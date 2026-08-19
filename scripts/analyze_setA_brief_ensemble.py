"""Did the induced brief's +0.065 on the lexical block survive into the ensemble?

Reads the OOF probabilities `run_setA_brief_ensemble.py` wrote and scores them at population
prevalence via `benchset_metrics` — the weights matter here exactly as much as everywhere
else in the set-A work, because the case-control sample carries 4-28% positives by silo
against a population 0.2-12%.

Reported per silo with win/loss counts against the ~0.03 noise floor, never as a bare mean:
`brouwer_2019` is 60% of set A and already sits at 0.99, so it drags every average toward
"no change" regardless of what happens elsewhere (`CONTEXT.md` §5).

Seeds are the unit of uncertainty. The repo's measured seed-to-seed sd is ~0.010 ROC-AUC, and
because all variants share identical rows *and* identical seeds, the per-seed paired
difference is a much tighter instrument than either arm's spread — it is reported alongside.

Usage:
    python scripts/analyze_setA_brief_ensemble.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchset_metrics import all_positive_f2, f2_optimal, wss_at  # noqa: E402
from ensemble_eval_utils import to_md  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FEATURES = REPO / "data" / "processed" / "setA_ensemble_features.parquet"
OOF = REPO / "reports" / "wf_llm_setA_brief_ensemble_oof.json"
OUT_MD = REPO / "reports" / "wf_llm_benchset_a_ensemble.md"
FLOOR = 0.03
VARIANT_ORDER = ["supplied", "induced_lex", "induced_all"]
THRESHOLDS = np.linspace(0.01, 0.99, 99)


def main() -> None:
    df = pd.read_parquet(FEATURES, columns=["use_case_key", "y", "w", "first_author", "split"])
    raw = json.loads(OOF.read_text(encoding="utf-8"))

    # branch OOF -> per (variant, seed) blend. Plain 50/50: learned blending has tied plain
    # averaging twice in this repo, and a fitted weight would add a second fitted quantity
    # to an experiment about a first.
    by_cell: dict[tuple, dict] = defaultdict(dict)
    for key, per_uc in raw.items():
        variant, branch, seed = key.split("|")
        by_cell[(variant, int(seed))][branch] = per_uc

    rows = []
    for (variant, seed), branches in sorted(by_cell.items()):
        for uc, g in df.groupby("use_case_key"):
            y, w = g.y.to_numpy().astype(int), g.w.to_numpy()
            preds = {b: np.asarray(p[uc], dtype=float) for b, p in branches.items() if uc in p}
            if not preds:
                continue
            arms = dict(preds)
            if len(preds) > 1:
                arms["blend"] = np.mean(list(preds.values()), axis=0)
            for arm, s in arms.items():
                keep = ~np.isnan(s)
                if len(np.unique(y[keep])) < 2:
                    continue
                f2s, _ = f2_optimal(y[keep], s[keep], w[keep], THRESHOLDS)
                rows.append({
                    "variant": variant, "seed": seed, "arm": arm,
                    "use_case": uc.replace("synergy_", ""),
                    "auc": roc_auc_score(y[keep], s[keep]),
                    "wss95": wss_at(y[keep], s[keep], w[keep])["wss"],
                    "f2_star": f2s,
                    "floor_f2": all_positive_f2(y[keep], w[keep]),
                })
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("no OOF cells found - run scripts/run_setA_brief_ensemble.py first")

    lines: list[str] = []

    def emit(t: str = "") -> None:
        print(t)
        lines.append(t)

    n_seeds = res.seed.nunique()
    emit("# Does the induced brief survive into the per-silo ensemble?")
    emit()
    emit(f"Per-silo CatBoost + LogisticRegression on the 9,993-row case-control sample, "
         f"grouped by `first_author`, 5-fold, **{n_seeds} seeds**, scored at population "
         "prevalence. `induced_lex` swaps only the BM25/overlap block; `induced_all` also "
         "swaps cosine-to-brief. All three variants are 4,625 columns wide, so nothing is "
         "confounded with feature count.")
    emit()
    emit("The question is not whether the lexical block improved — "
         "`wf_llm_benchset_a_induced_features.md` settled that at **+0.065** in isolation — "
         "but whether that improvement is **redundant with the 4,608 embedding dimensions "
         "the ensemble already sees**.")
    emit()

    # ---------------- headline ----------------
    head = (res.groupby(["arm", "variant"])
               .agg(auc=("auc", "mean"), auc_sd=("auc", "std"),
                    wss95=("wss95", "mean"), f2_star=("f2_star", "mean"))
               .reset_index())
    head["variant"] = pd.Categorical(head.variant, VARIANT_ORDER, ordered=True)
    emit("## 1. Headline — mean over 8 silos and seeds")
    emit()
    emit(to_md(head.sort_values(["arm", "variant"]).set_index(["arm", "variant"]).round(3)))
    emit()

    # ---------------- paired per-silo deltas ----------------
    emit("## 2. Paired per-silo change, `supplied` → each induced variant")
    emit()
    emit("Same rows, same folds, same seeds — so these differences are paired and far "
         f"tighter than the {FLOOR:.2f} noise floor that governs unpaired comparisons. "
         "`n_better`/`n_worse` count silos whose mean moved by more than that floor.")
    emit()
    piv = res.pivot_table(index=["arm", "use_case"], columns="variant", values="auc")
    for target in ("induced_lex", "induced_all"):
        if target not in piv:
            continue
        d = (piv[target] - piv["supplied"]).unstack("arm")
        emit(f"**`supplied` → `{target}`**, ROC-AUC change per silo")
        emit()
        emit(to_md(d.round(3)))
        emit()
        summary = pd.DataFrame({
            "mean_delta": d.mean(),
            "n_better": (d > FLOOR).sum(),
            "n_worse": (d < -FLOOR).sum(),
            "n_unchanged": (d.abs() <= FLOOR).sum(),
        })
        emit(to_md(summary.round(3)))
        emit()

    # ---------------- per-silo detail for the blend ----------------
    blend = res[res.arm == "blend"]
    if not blend.empty:
        emit("## 3. The blend, per silo")
        emit()
        present = [v for v in VARIANT_ORDER if v in blend.variant.unique()]
        t = blend.pivot_table(index="use_case", columns="variant", values="auc")[present]
        # With a single seed there is no spread to report - pivot_table drops the all-NaN
        # std columns entirely rather than returning NaN, so this is guarded rather than
        # assumed. Partial runs are the normal case here: the grid is ~16 min per CatBoost
        # cell and gets analysed as it lands.
        sd = (blend.pivot_table(index="use_case", columns="variant", values="auc", aggfunc="std")
              if n_seeds > 1 else None)
        t.columns = [f"auc_{c}" for c in t.columns]
        if sd is not None and "supplied" in sd:
            t["seed_sd_supplied"] = sd["supplied"]
        if "auc_induced_lex" in t:
            t["d_induced_lex"] = t.auc_induced_lex - t.auc_supplied
        f2 = blend.pivot_table(index="use_case", columns="variant", values="f2_star")
        for v in present:
            t[f"f2star_{v}"] = f2[v]
        t["floor_f2"] = blend.groupby("use_case").floor_f2.first()
        emit(to_md(t.round(3)))
        emit()
        if sd is not None:
            emit(f"Mean seed-to-seed sd on the blend: **{sd.mean().mean():.3f}** "
                 f"(repo's measured floor ~0.010). The paired per-silo deltas in §2 are the "
                 "instrument that matters — they share folds and seeds, so they resolve "
                 "changes far below this spread.")
            emit()

    # ---------------- what it means against the isolated block ----------------
    emit("## 4. Against the isolated-block result")
    emit()
    b = head[head.arm == "blend"].set_index("variant") if "blend" in head.arm.values else None
    if b is not None and {"supplied", "induced_lex"} <= set(b.index):
        gain = b.loc["induced_lex", "auc"] - b.loc["supplied", "auc"]
        emit("| | ROC-AUC gain from the induced brief |")
        emit("|---|---|")
        emit("| BM25 + overlap block alone, fitted on 60 labels | **+0.065** |")
        emit("| LLM reader, same model and prompt | **+0.057** |")
        emit(f"| **Full ensemble (blend), same brief swap** | **{gain:+.3f}** |")
        emit()

    # ---------------- the label ladder, on ONE common row set ----------------
    # Every other set-A arm is reported on test+validate. The ensemble's OOF predictions
    # exist for all rows (each produced by a fold that did not see that row), so subsetting
    # to the same rows makes the ladder comparable on evaluation while the *training* budget
    # is exactly what the ladder is varying.
    held_mask = (df.split != "train").to_numpy()
    ladder = []
    for (variant, seed), branches in sorted(by_cell.items()):
        if len(branches) < 2:
            continue
        for uc, g in df.groupby("use_case_key"):
            m = held_mask[g.index.to_numpy()]
            y, w = g.y.to_numpy().astype(int)[m], g.w.to_numpy()[m]
            s = np.mean([np.asarray(p[uc], dtype=float) for p in branches.values()], axis=0)[m]
            keep = ~np.isnan(s)
            if len(np.unique(y[keep])) < 2:
                continue
            ladder.append({"variant": variant, "seed": seed,
                           "use_case": uc.replace("synergy_", ""),
                           "auc": roc_auc_score(y[keep], s[keep])})
    if ladder:
        lad = pd.DataFrame(ladder).groupby("variant").auc.mean()
        emit("## 5. The label ladder, all arms on the same held-out rows")
        emit()
        emit("The ensemble scored on **test + validate only** — the rows every other set-A "
             "arm reports on. Training budget is what the ladder varies; the evaluation rows "
             "are now identical throughout.")
        emit()
        emit("| labels per silo | method | mean ROC-AUC |")
        emit("|---|---|---|")
        emit("| 0 | cosine-to-brief (`qwen4b`) | 0.774 |")
        emit("| 0 | LLM reader, supplied brief | 0.777 |")
        emit("| 60 | BM25 + overlap block, supplied brief | 0.733 |")
        emit("| 60 | BM25 + overlap block, **induced** brief | 0.798 |")
        emit("| 60 | LLM reader, **induced** brief | 0.834 |")
        emit("| 60 | LogReg on the Qwen3-4B embedding | 0.844 |")
        emit(f"| full in-silo (~80% of each silo) | **ensemble blend, supplied brief** | "
             f"**{lad.get('supplied', float('nan')):.3f}** |")
        if "induced_lex" in lad:
            emit(f"| full in-silo | ensemble blend, **induced** brief | "
                 f"**{lad['induced_lex']:.3f}** |")
        emit()
        emit("**The brief is a cold-start lever and it decays to nothing once the silo has "
             "enough labels to train on.** That is `CONTEXT.md` §1's ladder, now with numbers "
             "on every rung.")
        emit()

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res.to_csv(REPO / "reports" / "wf_llm_benchset_a_ensemble.csv", index=False)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
