"""What is each use-case brief field worth, and what do the classic bad specs actually cost?

Everything this repo knows about brief quality is about *replacing a whole brief* — the shuffled
control (`run_tier1b_control.py`), the induced rule set (`compare_setA_induced_brief.py`). Both
answer "does the brief matter". Neither answers the question a person filling in a form has:

    which of these fields carries the weight, and what happens if I write it badly?

`S-UCQ` (`academic_agent: spikes/use_case_quality/PLAN.md`) covers *which examples* and *induced
vs written rules*. It does not cover field-level value or degradation modes. This does.

Two arms, deliberately, because they answer different halves and can disagree:

  - **unfitted `bm25_nice` / `overlap_must_frac`** — one feature, no labels, no fitting. This is
    the cold-start rung and it reads the brief text directly, so it measures **spec quality**
    with nothing able to compensate for it. §5b of `wf_llm_benchset_a_findings.md` measured the
    largest single-feature brief effect here (+0.091).
  - **fitted lexical block, 30 pos + 30 neg** — the same 22 columns the ensemble's lexical branch
    sees, fitted on the same 60-label budget as the induced-brief ladder. A fitted model can
    *reweight around* a bad field, so this measures **what survives contact with labels**.

The gap between the two arms is the useful part: a field that only hurts the unfitted arm is a
cold-start problem that labels fix. A field that hurts both is a real spec defect.

Variants, and what each one is imitating
----------------------------------------
Ablations answer "what is this field worth": each brief field emptied in turn, everything else
held. Degradations answer "what does writing it badly cost", and each is a named failure mode:

  `only_name`      the 2-4 word use-case name and nothing else. This is not hypothetical - it is
                   what `embedding_utils.get_use_case_text` falls back to, so it is the floor the
                   `scripts/*.py` path actually ships (`CONTEXT.md` §3).
  `fluff_replace`  objective and problem_statement replaced by generic academic prose.
  `fluff_added`    the same prose *appended* to the real fields - dilution, not replacement, which
                   is the far more common way a real spec goes wrong.
  `conflicting`    `terms_exclude` moved into `terms_must_include`, so the spec demands exactly
                   what it should reject. This is S-AL's measured failure reproduced deliberately:
                   a hand-written policy scored 0.772 against 0.828 for no policy because it named
                   as a positive signal the category its own labels rejected.
  `keyword_flood`  must/nice padded with S-FF's measured generic vocabulary (`used`, `using`,
                   `analysis`, `data`, ...), whose median gold lift is 1.20x against 5.94x for the
                   same vocabulary ordered by signal.
  `must_only_one`  `terms_must_include` truncated to a single term - the "too few keywords" case.
  `vague_objective` objective replaced by a one-line abstraction that states an interest without
                   naming anything selectable.

Instrument check (ADR 0009, and it runs before any verdict): the rebuilt `full` variant is
correlated against the shipped `lex_*` columns. `compare_setA_induced_brief.py` reached 0.9993
this way and anything materially below that means the rebuild is not measuring the shipped block.
Note the IDF here is computed over the case-control sample rather than the full pool, so a small
gap is expected and is constant across variants - which is what makes variant-vs-variant valid
even where variant-vs-shipped is not.

Two surfaces
------------
`--set a` is the original run (8 SYNERGY collections, 2.19% positive) and keeps the original
output filenames, so its committed report and CSV reproduce unchanged. `--set b` is the clean
confirmatory surface (7 collections, 1.87% positive) and writes `*_set_b.*`.

Set B exists here for one specific reason: `wf_spec_quality_plan.md`'s `P-R` probe compares a
**reader** delta against a **matcher** delta, and the matcher's numbers were only ever measured
on set A. Subtracting a set-B reader delta from a set-A matcher delta would be comparing two
surfaces and calling the difference an arm effect — the selection-on-holdout error in miniature.
This re-run is free, so there is no reason to accept that.

Expect fewer collections in the *fitted* arm on B than on A: the arm needs 30 train positives
and 30 train negatives per collection, and B's smaller collections do not all have them. The
report prints which ones dropped out and why, because a win count over an unstated denominator
is the thing `CONTEXT.md` §5 exists to prevent.

    python scripts/run_spec_quality_ablation.py --seeds 5
    python scripts/run_spec_quality_ablation.py --seeds 5 --set b --drop-ab-crossing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from benchset_loader import SETS, case_control_sample, drop_ab_crossing, load_set  # noqa: E402
from ensemble_eval_utils import to_md  # noqa: E402
from lexical_features import build_lexical_features  # noqa: E402
from run_ensemble_candidate import logreg_c_fn  # noqa: E402

OUT_MD = REPO / "reports" / "wf_spec_quality_ablation.md"
OUT_CSV = REPO / "reports" / "wf_spec_quality_ablation.csv"
FIG = REPO / "reports" / "wf_spec_quality_ablation.png"


def out_paths(set_name: str) -> tuple[Path, Path, Path]:
    """Set A keeps the original filenames; any other set is suffixed so it cannot overwrite it."""
    stem = "wf_spec_quality_ablation" + ("" if set_name == "a" else f"_set_{set_name}")
    d = REPO / "reports"
    return d / f"{stem}.md", d / f"{stem}.csv", d / f"{stem}.png"

LOGREG_C = 0.0005
N_POS = N_NEG = 30  # the induced-brief ladder's budget, so this is comparable to it

FLUFF = (
    "This work investigates important aspects of the field and makes a significant contribution "
    "to the existing literature. A range of relevant factors is considered and the results are "
    "discussed in detail. The findings have implications for practice and further research is "
    "needed to explore these issues more fully."
)
VAGUE = "Find interesting and relevant recent work in this area that could be useful to us."
# S-FF Q3b: the most frequent top-document-frequency terms across its 28 pools.
GENERIC = ["used", "using", "analysis", "data", "results", "significant", "study", "based",
           "approach", "effects", "factors", "use", "different", "higher", "important"]

TERM_COLS = ["terms_must_include", "terms_nice_to_have", "terms_exclude"]
TEXT_FIELDS = ["objective", "problem_statement", "domain_industry", "domain_application"]


def _lst(v) -> list[str]:
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x) for x in v if x is not None and str(x).strip()]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    return [str(v)] if str(v).strip() else []


def variant(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """`df` with its brief columns rewritten. Never mutates the input."""
    d = df.copy()
    empty_list = [np.array([], dtype=object)] * len(d)

    def clear(col: str) -> None:
        d[col] = empty_list if col in TERM_COLS else ""

    if name == "full":
        pass
    elif name == "no_obj":
        clear("objective")
    elif name == "no_prob":
        clear("problem_statement")
    elif name == "no_must":
        clear("terms_must_include")
    elif name == "no_nice":
        clear("terms_nice_to_have")
    elif name == "no_exclude":
        clear("terms_exclude")
    elif name == "no_domain":
        for c in ("domain_industry", "domain_application"):
            clear(c)
        d["domain_technology_focus"] = empty_list
    elif name == "only_name":
        for c in TEXT_FIELDS:
            clear(c)
        for c in TERM_COLS:
            clear(c)
        # the name is what survives; it reaches the block through `dom`
        d["domain_technology_focus"] = empty_list
        d["domain_application"] = d["use_case_name"]
    elif name == "fluff_replace":
        d["objective"] = FLUFF
        d["problem_statement"] = FLUFF
    elif name == "fluff_added":
        d["objective"] = d["objective"].astype(str) + " " + FLUFF
        d["problem_statement"] = d["problem_statement"].astype(str) + " " + FLUFF
    elif name == "vague_objective":
        d["objective"] = VAGUE
    elif name == "conflicting":
        d["terms_must_include"] = [
            np.array(_lst(m) + _lst(e), dtype=object)
            for m, e in zip(d.terms_must_include, d.terms_exclude, strict=True)
        ]
        d["terms_exclude"] = empty_list
    elif name == "keyword_flood":
        d["terms_must_include"] = [np.array(_lst(m) + GENERIC, dtype=object)
                                   for m in d.terms_must_include]
        d["terms_nice_to_have"] = [np.array(_lst(n) + GENERIC, dtype=object)
                                   for n in d.terms_nice_to_have]
    elif name == "must_only_one":
        d["terms_must_include"] = [np.array(_lst(m)[:1], dtype=object)
                                   for m in d.terms_must_include]
    else:
        raise ValueError(name)
    return d


VARIANTS = ["full", "no_obj", "no_prob", "no_must", "no_nice", "no_exclude", "no_domain",
            "only_name", "fluff_replace", "fluff_added", "vague_objective", "conflicting",
            "keyword_flood", "must_only_one"]

UNFITTED = ["bm25_nice", "bm25_obj", "overlap_must_frac"]


def evaluate(sample: pd.DataFrame, seeds: int) -> pd.DataFrame:
    rows = []
    blocks: dict[str, pd.DataFrame] = {}
    for name in VARIANTS:
        blocks[name] = build_lexical_features(variant(sample, name))
        print(f"  built {name}", flush=True)

    feat_cols = [c for c in blocks["full"].columns if c != "has_exclude_terms"]

    for name, block in blocks.items():
        for uc, g in sample.groupby("use_case_key"):
            idx = g.index
            held = g.split != "train"
            y_ev = g.loc[held, "y"].to_numpy(float)
            if y_ev.sum() < 5:
                continue
            b = block.loc[idx]

            rec = {"variant": name, "use_case": uc}
            for col in UNFITTED:
                s = b.loc[held.to_numpy(), col]
                # Emptying a brief field makes its feature *uncomputable*, and the module emits
                # NaN + an indicator rather than 0 (NULL is not 0). So the honest report is
                # "undefined", never an imputed AUC - the absence is the finding.
                rec[f"unfitted_{col}"] = (
                    np.nan if s.isna().any() else roc_auc_score(y_ev, s)
                )

            pos = g[(g.split == "train") & (g.y == 1)]
            neg = g[(g.split == "train") & (g.y == 0)]
            if len(pos) < N_POS or len(neg) < N_NEG:
                rec["fitted_auc"] = np.nan
            else:
                aucs = []
                for seed in range(seeds):
                    rng = np.random.default_rng(1000 * seed + 7)
                    tr = np.concatenate([
                        pos.index[rng.choice(len(pos), N_POS, replace=False)].to_numpy(),
                        neg.index[rng.choice(len(neg), N_NEG, replace=False)].to_numpy(),
                    ])
                    pipe = logreg_c_fn(feat_cols, C=LOGREG_C)()
                    pipe.fit(b.loc[tr, feat_cols], sample.loc[tr, "y"].to_numpy())
                    s = pipe.predict_proba(b.loc[held.to_numpy(), feat_cols])[:, 1]
                    aucs.append(roc_auc_score(y_ev, s))
                rec["fitted_auc"] = float(np.mean(aucs))
            rows.append(rec)
    return pd.DataFrame(rows)


def chart(summary: pd.DataFrame, fig: Path, surface: str, n_coll: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary.drop(index="full").sort_values("d_fitted_auc")
    yy = np.arange(len(s))
    f, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    ax = axes[0]
    series = [("d_unfitted_bm25_obj", "bm25 vs objective"),
              ("d_unfitted_bm25_nice", "bm25 vs nice-to-have"),
              ("d_unfitted_overlap_must_frac", "term overlap vs must-include")]
    for i, (col, lab) in enumerate(series):
        ax.barh(yy + (i - 1) * 0.27, s[col].fillna(0), height=0.26, label=lab)
    ax.set_title("Cold start — 0 labels, one feature, nothing able to compensate")
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    ax.barh(yy, s.d_fitted_auc, height=0.6, color="tab:red")
    ax.set_title("After 60 labels — the full 22-column lexical block, fitted")

    for ax in axes:
        ax.axvline(0, color="k", lw=1)
        ax.axvspan(-0.03, 0.03, color="grey", alpha=0.18)
        ax.set_xlabel("ROC-AUC change vs the full shipped brief")
        ax.grid(alpha=0.3, axis="x")
    axes[0].set_yticks(yy); axes[0].set_yticklabels(s.index)
    f.suptitle("What each brief field is worth, and what writing it badly costs — "
               f"{surface}, {n_coll} collections\n"
               "negative is worse than the full brief; a bar at 0 means that variant does not "
               "touch that feature (NaN = the feature becomes uncomputable). The grey band is "
               "±0.03, the smallest gap this repo treats as real — a bar inside it is noise.",
               fontsize=10)
    f.tight_layout(); f.savefig(fig, dpi=130)
    print(f"wrote {fig}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--set", default="a", choices=sorted(SETS),
                    help="which large split to score; 'a' keeps the original output filenames")
    ap.add_argument("--drop-ab-crossing", action="store_true",
                    help="exclude papers that also appear in the other large split, for a strict "
                         "held-out read (154 papers straddle the A/B boundary)")
    args = ap.parse_args()

    out_md, out_csv, fig = out_paths(args.set)
    surface = f"benchset_v1_large_set_{args.set}"

    base = load_set(args.set)
    # No `make_splits` here: `case_control_sample` computes `split` on the sample it returns, so
    # assigning it on the 62k/82k-row frame first was a discarded fold computation over the whole
    # corpus. Removing it changes no output - the sample's column is overwritten either way.
    n_before = len(base)
    if args.drop_ab_crossing:
        base = drop_ab_crossing(base, args.set)
        print(f"strict held-out: dropped {n_before - len(base)} rows that also appear in the "
              f"other large split", flush=True)
    sample = case_control_sample(base).reset_index(drop=True)
    print(f"sample: {len(sample)} rows, {sample.use_case_key.nunique()} collections", flush=True)

    res = evaluate(sample, args.seeds)
    res.to_csv(out_csv, index=False)

    metrics = [c for c in res.columns if c.startswith(("unfitted_", "fitted_"))]
    summary = res.groupby("variant")[metrics].mean().loc[VARIANTS]
    for m in metrics:
        summary[f"d_{m}"] = summary[m] - summary.loc["full", m]

    # win/loss counts against `full`, per collection — CONTEXT.md §5
    piv = res.pivot(index="use_case", columns="variant", values="fitted_auc")
    worse = {v: int((piv[v] < piv["full"] - 0.03).sum()) for v in VARIANTS}
    pivu = res.pivot(index="use_case", columns="variant", values="unfitted_bm25_nice")
    worse_u = {v: int((pivu[v] < pivu["full"] - 0.03).sum()) for v in VARIANTS}
    summary["collections_worse_fitted"] = pd.Series(worse)
    summary["collections_worse_unfitted"] = pd.Series(worse_u)

    # The denominators, stated rather than implied. The fitted arm needs 30 train positives and 30
    # train negatives per collection; a collection short of either is NaN, not a zero, and it must
    # not be counted in a "worse on N of M" claim (NULL is not 0).
    n_coll = int(res.use_case.nunique())
    n_fitted = int(piv["full"].notna().sum())
    dropped = sorted(piv.index[piv["full"].isna()])

    chart(summary, fig, f"set {args.set.upper()}", n_coll)

    shipped = sample[["lex_bm25_obj", "lex_bm25_nice", "lex_overlap_must_frac"]]
    rebuilt = build_lexical_features(variant(sample, "full"))
    checks = {c.replace("lex_", ""): float(
        np.corrcoef(shipped[c].fillna(0), rebuilt[c.replace("lex_", "")].fillna(0))[0, 1]
    ) for c in shipped.columns}

    cols = ["unfitted_bm25_nice", "d_unfitted_bm25_nice", "collections_worse_unfitted",
            "fitted_auc", "d_fitted_auc", "collections_worse_fitted"]
    strict = (" Papers that also appear in the other large split are excluded, so this is a "
              "strict held-out read." if args.drop_ab_crossing else "")
    drop_note = (
        f"\n\n⚠️ **The fitted arm covers {n_fitted} of the {n_coll} collections, not all "
        f"{n_coll}.** {', '.join(f'`{d}`' for d in dropped)} lack the 30 train positives + 30 "
        "train negatives the arm needs, so their score is **undefined rather than zero** and they "
        "are excluded from `collections_worse_fitted`. Read that count against "
        f"{n_fitted}, not {n_coll}.\n" if dropped else "\n"
    )
    out_md.write_text(
        "# Use-case spec quality — what each field is worth, and what bad writing costs\n\n"
        f"Generated by `scripts/run_spec_quality_ablation.py --seeds {args.seeds} --set "
        f"{args.set}` on `{surface}` ({n_coll} collections). Held-out rows only.{strict} Read the "
        "script docstring for what each variant imitates.\n\n"
        "**Instrument check (ADR 0009, run before any verdict).** Rebuilt `full` variant against "
        "the shipped `lex_*` columns: "
        + ", ".join(f"`{k}` r={v:.4f}" for k, v in checks.items())
        + ". IDF here is computed over the case-control sample rather than the full pool, so a "
        "gap against the shipped column is expected; it is constant across variants, which is "
        "what makes variant-vs-variant comparison valid.\n\n"
        "## 0. How to read any number here\n\n"
        "Every score is **ROC-AUC**: the chance that this feature ranks a genuinely relevant "
        "paper above an irrelevant one. **0.500 is a coin flip** and 1.000 is perfect, so a "
        "number below 0.500 means the feature is pointing the wrong way — worse than useless.\n\n"
        "`d_` columns are the change against the **full shipped brief**, so they answer *what did "
        "damaging the brief this way cost?* **Negative is worse.** A `d_` of 0.000 usually means "
        "the variant does not touch that particular feature at all, not that the damage was free.\n\n"
        "**A gap smaller than 0.03 is not a result.** Re-running with different random seeds moves "
        "these numbers by about 0.010 on its own (`CONTEXT.md` §5), so anything under ~0.03 is "
        "inside the noise. That is why `collections_worse_*` sits beside every mean: it counts how "
        f"many of the {n_coll} collections fall more than 0.03 below `full`. A mean can be dragged "
        "by one easy collection; a count cannot.\n\n"
        "Blank (`NaN`) means **uncomputable, not zero**. Emptying a brief field can leave its "
        "feature with nothing to read, and the honest report of that is an absence — which is "
        "itself the finding."
        + drop_note + "\n"
        "## 1. Results\n\n"
        + to_md(summary[cols].round(4)) + "\n\n"
        "## 2. All metrics\n\n" + to_md(summary.round(4)) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
