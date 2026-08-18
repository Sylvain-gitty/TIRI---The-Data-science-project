"""Read the reader arm against the matcher arm, and read `P-R`'s three bars.

Companion to `run_spec_quality_reader.py`, which buys the responses. This spends nothing.

The comparator problem, and why this reports two of them
-------------------------------------------------------
`P-R`'s H1 and H3 are comparisons *between arms*: does damaging the brief this way cost a
**reader** more (or less) than it cost a **matcher**? That needs a matcher number to subtract,
and the matcher produced two, at different label counts:

  - the **fitted** arm (60 labels, the full 22-column lexical block). This is the comparator the
    plan names, quoting -0.014 for `conflicting` and -0.072 for `keyword_flood`.
  - the **unfitted** arm (0 labels, one feature, nothing able to compensate). This is the one
    actually *label-matched* to a zero-shot reader.

Neither is a clean match, and the honest response is to print both rather than to pick the
flattering one. The subtlety that makes the unfitted arm easy to misuse: **each variant damages a
different feature.** `conflicting` moves terms between must-include and exclude, so it touches
`overlap_must_frac` and leaves `bm25_obj` and `bm25_nice` at exactly 0.000. `fluff_replace`
rewrites prose, so it touches `bm25_obj` alone (-0.177 on set A, -0.169 on set B) and nothing
else. Reading one fixed unfitted column for every variant would therefore report a confident
0.000 for variants it simply does not measure. So the unfitted comparator here is the **most
negative** of the three - "the most the matcher noticed at zero labels" - and the per-feature
detail is printed beside it.

The bars are read against the fitted arm, because that is what the plan pre-registered. Where the
reader delta clears **both** comparators the conclusion is robust to the choice, and that is said
explicitly; where it clears only one, that is said too.

Weights
-------
`w` is joined back by rebuilding the same seeded sample `run_spec_quality_reader.build_sample`
produced. AUC is unbiased unweighted on a case-control sample, but F2@own, precision and
fraction-read are not - they need `w` or they are flatteringly wrong (`benchset_loader`'s
docstring spells out which is which). Importing `build_sample` rather than re-deriving it is the
same one-function-both-arms discipline the probe itself rests on.

    python scripts/analyze_spec_quality_reader.py --stage 1
    python scripts/analyze_spec_quality_reader.py --stage 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analyze_benchset_a import cell_metrics  # noqa: E402
from ensemble_eval_utils import to_md  # noqa: E402
from run_spec_quality_reader import build_sample  # noqa: E402

REPORTS = REPO / "reports"
FLOOR = 0.03  # CONTEXT.md §5: gaps under this are not established
NEED = 5      # the pre-registered win count: ">=5 of 7" on set B, ">=5 of 8" on set A
UNFITTED = ["unfitted_bm25_obj", "unfitted_bm25_nice", "unfitted_overlap_must_frac"]


def matcher_deltas(set_name: str) -> pd.DataFrame:
    """Per (variant, collection) matcher deltas against `full`, both arms.

    `unfitted_worst` is the most negative of the three single-feature deltas - see the module
    docstring on why a fixed column would report 0.000 for variants it does not measure.
    """
    stem = "wf_spec_quality_ablation" + ("" if set_name == "a" else f"_set_{set_name}")
    d = pd.read_csv(REPORTS / f"{stem}.csv")
    d["use_case"] = d.use_case.str.replace("synergy_", "", regex=False)
    out = []
    for uc, g in d.groupby("use_case"):
        base = g[g.variant == "full"].iloc[0]
        for _, r in g.iterrows():
            unf = {c: r[c] - base[c] for c in UNFITTED}
            out.append({
                "use_case": uc, "variant": r.variant,
                "matcher_fitted_d": r.fitted_auc - base.fitted_auc,
                "matcher_unfitted_worst_d": min(unf.values()) if not any(
                    np.isnan(v) for v in unf.values()) else np.nan,
                **{f"matcher_d_{c.replace('unfitted_', '')}": unf[c] for c in UNFITTED},
            })
    return pd.DataFrame(out)


def reader_metrics(stage: int) -> pd.DataFrame:
    path = REPORTS / f"wf_spec_quality_reader_stage{stage}_responses.parquet"
    if not path.exists():
        raise SystemExit(f"{path} missing - run run_spec_quality_reader.py --stage {stage} --spend")
    resp = pd.read_parquet(path)

    sample, _ = build_sample(stage)
    keys = sample[["paper_id", "use_case_key", "w", "split"]]
    resp = resp.merge(keys, on=["paper_id", "use_case_key"], how="left", validate="many_to_one")
    if resp["w"].isna().any():
        raise RuntimeError("responses not covered by the rebuilt sample - the seed or the "
                           "sampler changed, so nothing here is comparable")

    rows = []
    for (v, uc), g in resp.groupby(["spec_variant", "use_case_key"]):
        m = cell_metrics(g)
        if m:
            rows.append({"variant": v, "use_case": uc.replace("synergy_", ""),
                         "parse_rate": float(g.parsed.mean()), **m})
    return pd.DataFrame(rows)


def deltas(rd: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = []
    for uc, g in rd.groupby("use_case"):
        if "full" not in set(g.variant):
            continue
        base = g[g.variant == "full"].iloc[0]
        for _, r in g.iterrows():
            out.append({"use_case": uc, "variant": r.variant,
                        **{f"reader_d_{c}": r[c] - base[c] for c in cols}})
    return pd.DataFrame(out)


def verdict(name: str, ok: bool, bar: str, evidence: str) -> str:
    return (f"**Pre-registered bar: {'PASS' if ok else 'FAIL'}.** {name} required {bar} "
            f"{evidence}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=2, choices=(1, 2))
    args = ap.parse_args()
    set_name = "a" if args.stage == 1 else "b"

    rd = reader_metrics(args.stage)
    md = matcher_deltas(set_name)
    rdd = deltas(rd, ["auc", "f2_own", "screened_own", "recall_own"])
    joined = rdd.merge(md, on=["use_case", "variant"], how="left")

    n_coll = rd.use_case.nunique()
    print(f"stage {args.stage} (set {set_name.upper()}): {n_coll} collections, "
          f"{rd.variant.nunique()} variants")

    per_variant = joined.groupby("variant").agg(
        reader_d_auc=("reader_d_auc", "mean"),
        reader_d_f2own=("reader_d_f2_own", "mean"),
        reader_d_read=("reader_d_screened_own", "mean"),
        matcher_fitted_d=("matcher_fitted_d", "mean"),
        matcher_unfitted_worst_d=("matcher_unfitted_worst_d", "mean"),
    )
    # Win counts, because a mean over heterogeneous collections can be carried by one of them.
    per_variant["reader_worse_n"] = joined.groupby("variant").apply(
        lambda g: int((g.reader_d_auc < -FLOOR).sum()), include_groups=False)
    per_variant["reader_beats_matcher_n"] = joined.groupby("variant").apply(
        lambda g: int((g.reader_d_auc < g.matcher_fitted_d - FLOOR).sum()), include_groups=False)

    hyp = []
    # The pre-registered count is the literal 5 of the amendment's ">=5 of 7" (and the original
    # ">=5 of 8"), NOT a computed majority - a majority of 7 would be 4, which would be quietly
    # relaxing a bar after the fact. Asserting the width stops the number being applied to a set
    # it was never registered against.
    need = NEED
    if n_coll not in (7, 8):
        raise SystemExit(f"{n_coll} collections: the bar was pre-registered as >={NEED} of 7 "
                         f"(set B) or >={NEED} of 8 (set A). Re-register before reading it here.")
    if "conflicting" in per_variant.index:
        r = per_variant.loc["conflicting"]
        ok = r.reader_beats_matcher_n >= need
        hyp.append(verdict(
            "H1 (contradiction costs a reader more than a matcher)", ok,
            f"`conflicting` to cost the reader >={FLOOR} more AUC than it cost the matcher on "
            f">={need} of {n_coll} collections.",
            f"Reader {r.reader_d_auc:+.3f} against matcher {r.matcher_fitted_d:+.3f} (fitted) / "
            f"{r.matcher_unfitted_worst_d:+.3f} (unfitted, worst feature); clears the gap on "
            f"{int(r.reader_beats_matcher_n)} of {n_coll}."))
    for v in ("fluff_replace", "vague_objective"):
        if v in per_variant.index:
            r = per_variant.loc[v]
            ok = r.reader_d_auc <= -FLOOR
            hyp.append(verdict(
                f"H2 (`{v}` damages a reader at every label count)", ok,
                f"`{v}` to cost the reader >={FLOOR} AUC.",
                f"Reader {r.reader_d_auc:+.3f}, worse than the floor on "
                f"{int(r.reader_worse_n)} of {n_coll} collections."))
    if "keyword_flood" in per_variant.index:
        r = per_variant.loc["keyword_flood"]
        ok = int((joined[joined.variant == "keyword_flood"].eval(
            "reader_d_auc > matcher_fitted_d")).sum()) >= need
        hyp.append(verdict(
            "H3 (term-list padding costs a reader LESS than a matcher)", ok,
            f"`keyword_flood` to cost the reader less than the matcher on >={need} of {n_coll}.",
            f"Reader {r.reader_d_auc:+.3f} against matcher {r.matcher_fitted_d:+.3f} (fitted)."))

    out_csv = REPORTS / f"wf_spec_quality_reader_stage{args.stage}.csv"
    joined.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    print("\n" + to_md(per_variant.round(4)))
    print()
    for h in hyp:
        print(h)
    print("\nper-collection reader metrics:")
    print(to_md(rd.set_index(["variant", "use_case"])[
        ["n", "auc", "f2_own", "recall_own", "screened_own", "tie_frac", "parse_rate"]].round(3)))


if __name__ == "__main__":
    main()
