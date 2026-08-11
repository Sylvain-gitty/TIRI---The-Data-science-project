"""analyze_axis_and_iterations.py — compares all CatBoost configs fetched so far
(screening, tuned=150/4, axis_add, axis_replace, iterations_300) across 5 seeds, F2 as the
primary metric (per instruction: "remember we care about F2"), ROC-AUC as secondary.
Picks the winner to carry into nested weight selection (#2) and SYNERGY validation (#1).

Usage:
    python scripts/analyze_axis_and_iterations.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_eval_utils import to_md  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "processed" / "papers_fe.parquet"
HPARAM_JSON = REPO / "reports" / "wf_ensemble_v2_hparam_oof.json"
AXIS_JSON = REPO / "reports" / "wf_ensemble_v2_axis_iterations_oof.json"
OUT_MD = REPO / "reports" / "wf_ensemble_v2_experiments.md"
NOISE_FLOOR = 0.03

USE_CASE_DISPLAY = {
    "carbon_capture": "Carbon Capture",
    "cement_binders": "Low-Carbon Cement",
    "solar_leo": "Solar Cells for Satellites",
    "soil_microbiome": "Soil Microbiome",
    "ner": "Named Entity Recognition",
    "tech_forecasting": "Technology Prediction",
}


def f2_at(y_true, y_score, t=0.5):
    return fbeta_score(y_true, (y_score >= t).astype(int), beta=2, zero_division=0)


def f2_optimal(y_true, y_score, thresholds=np.linspace(0.05, 0.95, 91)):
    best_t, best_f2 = 0.5, -1.0
    for t in thresholds:
        f2 = f2_at(y_true, y_score, t)
        if f2 > best_f2:
            best_t, best_f2 = float(t), float(f2)
    return best_t, best_f2


def main() -> None:
    hparam = json.load(open(HPARAM_JSON))
    axis = json.load(open(AXIS_JSON))

    df = pd.read_parquet(DATA_PATH)
    y = df["y"].to_numpy().astype(int)
    use_case = df["use_case_key"]
    USE_CASE_ORDER = sorted(use_case.unique())
    y_by_uc = {uc: y[(use_case == uc).to_numpy()] for uc in USE_CASE_ORDER}

    # Normalize both JSON shapes: hparam_oof.json is {config_name: {uc: [floats]}} for a
    # single (already-committed) seed structure per earlier round; axis_iterations_oof.json
    # is {key: {seed_str: {uc: [floats]}}}. Reshape everything to
    # configs[name][seed][uc] -> np.array for a uniform loop below.
    configs: dict[str, dict[int, dict]] = {}

    # hparam_oof.json only has a single (seed=0) dict per key - run_multiseed_verification.py
    # already fetched 5 seeds of screening/tuned for its own analysis but never persisted the
    # raw OOF arrays, only summary deltas. Only "tuned_150_4" is needed here as the baseline
    # this round compares against - not "screening", already established as worse.
    import modal

    fn = modal.Function.from_name("tiri-ensemble-ablation", "run_catboost_hparam_oof")
    print("Re-fetching tuned (150/4) OOF at 5 seeds as this round's baseline ...")
    configs["tuned_150_4"] = {}
    results = fn.starmap([(150, 4, s) for s in range(5)])
    for seed, res in enumerate(results):
        configs["tuned_150_4"][seed] = {uc: np.array(v) for uc, v in res.items()}
    print("  refetched tuned_150_4")

    for key in ("axis_add", "axis_replace", "iterations_300"):
        configs[key] = {}
        for seed_str, res in axis[key].items():
            configs[key][int(seed_str)] = {uc: np.array(v) for uc, v in res.items()}

    # ---- Score every config, every seed, every use case -------------------------------
    rows = []
    for config_name, per_seed in configs.items():
        for seed, oof_by_uc in per_seed.items():
            for uc in USE_CASE_ORDER:
                yu = y_by_uc[uc]
                oof = oof_by_uc[uc]
                t_star, f2_star = f2_optimal(yu, oof)
                rows.append({
                    "config": config_name, "seed": seed, "use_case": USE_CASE_DISPLAY[uc],
                    "roc_auc": roc_auc_score(yu, oof),
                    "f2_at_0.5": f2_at(yu, oof, 0.5),
                    "f2_at_t_star": f2_star,
                })
    all_df = pd.DataFrame(rows)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("## 7. Axis feature (PCA/SVD follow-up) + iterations=300, all vs. the tuned baseline")
    emit()
    emit("All configs at 5 seeds. F2@t* is the headline metric (per-use-case optimal "
         f"threshold); ROC-AUC secondary. Noise floor: ~{NOISE_FLOOR:.2f} (CONTEXT.md §5).")
    emit()

    summary = all_df.groupby("config")[["roc_auc", "f2_at_0.5", "f2_at_t_star"]].mean().round(4)
    order = ["tuned_150_4", "iterations_300", "axis_add", "axis_replace"]
    summary = summary.reindex([o for o in order if o in summary.index])
    emit(to_md(summary, "config"))
    emit()

    baseline = "tuned_150_4"
    for config_name in order:
        if config_name == baseline or config_name not in configs:
            continue
        merged = all_df[all_df.config.isin([baseline, config_name])]
        piv = merged.pivot_table(index=["seed", "use_case"], columns="config", values="f2_at_t_star")
        delta = (piv[config_name] - piv[baseline])
        emit(f"**{config_name} vs. {baseline}**: mean F2@t* delta **{delta.mean():+.4f}** "
             f"(sd {delta.std():.4f}). Clears noise floor: "
             f"{'YES' if delta.mean() > NOISE_FLOOR else 'no'}.")
        emit()

    per_uc_detail = all_df.groupby(["config", "use_case"])["f2_at_t_star"].agg(["mean", "std"]).round(4)
    emit("Per-use-case F2@t*, mean +/- sd across 5 seeds:")
    emit()
    for config_name in order:
        if config_name not in configs:
            continue
        emit(f"*{config_name}*")
        emit()
        emit(to_md(per_uc_detail.loc[config_name].round(4), "use_case"))
        emit()

    existing = OUT_MD.read_text() if OUT_MD.exists() else ""
    OUT_MD.write_text(existing + "\n" + "\n".join(lines) + "\n")
    print(f"\nAppended to {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
