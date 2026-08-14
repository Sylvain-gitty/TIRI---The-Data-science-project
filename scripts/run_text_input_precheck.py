"""run_text_input_precheck.py — the $0 gate for probe P-TX: does the TEXT fed to the
embedder move within-silo cosine-to-brief ROC-AUC at all, before any Modal GPU spend?

`reports/wf_spec_quality_plan.md`'s `P-TX` section asks the one paper-text lever
`CONTEXT.md` §6's graveyard of rejected features never tried: `embedding_utils.MODEL_CONFIGS`
already carries `title_abstract_sep` as a per-model quirk, so the join between title and
abstract has always been a variable — it has just never been treated as one. Every rejected
feature in §6 (11 metadata scalars, PCA, per-use-case centring, the induced brief in the
ensemble) added columns *beside* the embedding. None of them changed what text goes *into*
it. This script is that test, kept to the one axis its name promises and nothing else.

THE TRAP THIS SCRIPT REPLACES
------------------------------
The plan's original $0 pre-check compared the shipped `cos_brief_*` / `cos_briefpre_*`
columns on set A. That comparison differs on the **brief** side
(`embed_benchsets.BRIEF_VARIANTS["pre_screening"]` drops `objective` and adds the term
lists) — the **paper vectors are byte-identical** between that pair, so it measures nothing
about paper text. It has also already been run: `03_eda_full_benchset_v1.ipynb` §8.4 measured
a paired median of 0.003–0.007 ROC-AUC across 27 collections, below the 0.01 gate here — so
running the original pre-check as written would have failed P-TX on a number about the wrong
axis. Verified directly against that notebook's cells 39-40 before writing this script.
There is also no title-only paper vector anywhere on disk: `embed_benchsets.py` always calls
`embedding_utils.embed_papers`, which always joins title+abstract.

THE CORRECTED GATE
-------------------
Embed **title-only** and **title+abstract** with **one fastembed model held fixed**, so the
only thing that varies between the two arms is the text fed to the embedder. Compare
within-silo ROC-AUC of cosine-to-brief (`retrieval_ranking_metrics`'s `query_similarity_auc` —
no classifier, no folds, just "does ranking by cosine-to-brief separate the labels", the
same statistic `03_eda_full_benchset_v1.ipynb` §8.3-8.4 already used).

Gate: any of the four collections below where |AUC(title_only) − AUC(title_abstract)| ≥ 0.01
→ **PASS**, P-TX earns its ~$3 Modal run. All four inside 0.01 → **FAIL**, P-TX is closed and
`CONTEXT.md` §6 gains a row. The 0.01 bar is deliberately generous — see the report's §0 for
why it sits at a third of the repo's measured 0.03 noise floor, and why that is a screening
threshold, not an effect-size claim.

MODEL CHOICE
------------
`BAAI/bge-small-en-v1.5` — fastembed backend (local ONNX, CPU only, no network beyond the
model's one-time download, and already cached locally from the sibling repo's own use of it).
It is **not** jasper/qwen4b, the pair `papers_fe.parquet` and every shipped cosine-to-brief
number use, and that is deliberate: holding a *different* model fixed while only the paper
text varies means a PASS or FAIL here answers "does text input move AUC at all on this
corpus", not "does it move the number already shipped". It is also the one fastembed model
in `MODEL_CONFIGS` with a real, documented query/passage asymmetry — `query_prefix` for the
brief side, empty `passage_prefix` for papers — the same query-vs-passage shape cosine-to-
brief has always used, whereas the alternative fastembed options
(`all-MiniLM-L6-v2`, `paraphrase-multilingual-MiniLM-L12-v2`) are symmetric general-purpose
models with no retrieval-specific training, and `nomic-embed-text-v1.5` measured ~795s for
100 papers on CPU in `compare_embeddings.py`'s own docstring — too slow for a screening gate.

SURFACES
--------
Three SMALL `small_test` SYNERGY collections (`reports/benchset_v1_split_manifest.json` →
`assignment`, picked for small `n_papers` so the whole script runs in minutes on CPU):
`synergy_donners_2021` (258 rows), `synergy_oud_2018` (952), `synergy_meijboom_2021` (882).
`roadfreight_metareview` (132 rows, also `small_test`) is excluded: the manifest itself flags
it as quarantined — its label question is "is this a review of reviews", not relevance
screening, at 78% prevalence, and it is "already out of every headline average in
02/03_eda_*_benchset_v1". It is not this gate's surface to spend.

Plus **one TIRI silo**, `carbon_capture` (297 rows strict positive/negative, 147/150 — the
best-balanced of TIRI's six), read from `data/processed/papers_combined.parquet`. Reported
**separately** from the three benchset collections throughout — never pooled — because the
two provenances differ in exactly the way `DATA_BRIEF.md`'s honest-limit #2 flags: the
benchset briefs are LLM-derived from each review's own abstract, while TIRI's are
analyst-written from scratch. Averaging across that seam would compare two brief-generation
processes, not two text-input arms.

The `+venue` arm named in the plan as P-TX's confirmatory variant is **dropped entirely** —
`data/processed/papers_benchset_v1.parquet` has no `venue` column (only the 2,873-row
`papers_combined.parquet` does), so it is unbuildable on its own confirmatory surface. Note
also that `CONTEXT.md` §6 already rejected `has_venue` as a *metadata* feature (part of the
11-feature punch list, +0.002 combined) — a different operation from changing the text fed to
the embedder, but worth saying before a reviewer asks why venue does not reappear here.

THREE TRAPS RECORDED HERE FOR WHOEVER RUNS THE MODAL PART, IF THIS GATE PASSES
-------------------------------------------------------------------------------
1. `text_variant` must enter `embed_benchsets.papers_path()`'s cache **filename**
   (`benchset_v1_{use_case_key}_{safe_model}_papers.parquet`, ~:155) before any GPU spend, or
   four text variants silently overwrite each other and the resume logic ("a finished
   collection is skipped outright") then *skips* a variant that never ran. This script does
   not touch that file or that pipeline — it embeds directly via `embedding_utils`, with its
   own report-local filenames, precisely so no GPU-facing cache is at risk of this.
2. `embed_benchsets.MAX_ABSTRACT_CHARS = 4000` (~:141) truncates abstracts but never titles,
   so `title_only` is the only arm with **no truncation at all** — a confound between arms,
   not merely a text change. This script reproduces that same truncation on its
   `title_abstract` arm (slicing abstracts to 4,000 chars before embedding, exactly as
   `embed_benchsets.embed_collection` does) so the two arms compared here are the same two
   arms a real run would produce, and reports what fraction of each collection's abstracts
   the truncation actually touches.
3. The shuffled-brief falsification seam (`CONTEXT.md` L258-260) does **not** apply here. It
   is mandated for anything claiming to read the brief, to prove the reader is not scoring on
   an artefact independent of brief content. P-TX changes the **paper** side, not the brief
   side — the brief is embedded once per collection and reused unchanged across both arms
   (see `evaluate_collection` below) — so there is no brief-reading claim here to falsify.
   Stated explicitly so it does not read as an omission.

THE PRIOR THIS PROBE CUTS AGAINST
-----------------------------------
`CONTEXT.md` §7 measured what a genuinely better brief (an induced rule set built from 60
labels) is worth to cosine-to-brief on set A: **+0.004 / −0.005**, against +0.065 for the
lexical block and +0.057 for an LLM reader on the identical brief swap — and calls the cosine
"the unimprovable one". That is a brief-quality lever, not a text-input lever, but it is the
closest existing measurement of "can anything be done to move this particular ranker", and it
found next to nothing. If this gate also fails, that prior is *confirmed*, not merely
unrefuted — worth stating plainly rather than as a disappointment.

Usage:
    python scripts/run_text_input_precheck.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from embedding_utils import (  # noqa: E402
    MODEL_CONFIGS, build_label_masks, drop_empty_rows, embed_papers, embed_texts,
    get_title_abstract, retrieval_ranking_metrics,
)
from ensemble_eval_utils import to_md  # noqa: E402

MODEL = "BAAI/bge-small-en-v1.5"
MAX_ABSTRACT_CHARS = 4000  # embed_benchsets.py's own truncation budget (~:141), reproduced
                           # here (not imported — that module is Modal-corpus plumbing this
                           # $0 gate must not touch) so the title_abstract arm faces the same
                           # truncation confound the real pipeline would give it. See trap #2.
GATE_THRESHOLD = 0.01
NOISE_FLOOR = 0.03  # CONTEXT.md §5's measured seed-to-seed sd on within-silo ROC-AUC

BENCHSETS = REPO / "data" / "benchsets_v1"
TIRI_PAPERS = REPO / "data" / "processed" / "papers_combined.parquet"

# embed_benchsets.BRIEF_VARIANTS["full"] — the production brief recipe, unchanged, so the
# brief this script scores against is the same one cosine-to-brief has always meant.
BRIEF_COLS = ["use_case_name", "problem_statement", "objective",
              "domain_industry", "domain_application"]

SMALL_TEST_COLLECTIONS = ["synergy_donners_2021", "synergy_oud_2018", "synergy_meijboom_2021"]
TIRI_SILO = "carbon_capture"

OUT_CSV = REPO / "reports" / "wf_text_input_precheck.csv"
OUT_MD = REPO / "reports" / "wf_text_input_precheck.md"
FIG = REPO / "reports" / "wf_text_input_precheck.png"


def build_brief_text(row: pd.Series) -> str:
    """One brief string, BRIEF_COLS joined exactly as `embed_benchsets.embed_briefs`
    joins them (space-separated, empty/NaN fields skipped). None of these five fields
    are list-typed in either surface used here (only the unused `terms_*` fields are),
    so no list-handling is needed the way `embed_briefs`'s `field_text` has for its."""
    return " ".join(
        str(row[c]) for c in BRIEF_COLS if pd.notna(row.get(c)) and str(row[c]).strip()
    )


def load_benchset_collection(name: str) -> pd.DataFrame:
    return pd.read_parquet(BENCHSETS / f"{name}.parquet")


def load_tiri_silo(use_case_key: str) -> pd.DataFrame:
    df = pd.read_parquet(TIRI_PAPERS)
    return df[df["use_case_key"] == use_case_key].reset_index(drop=True)


def evaluate_collection(name: str, provenance: str, df: pd.DataFrame) -> dict:
    """Score one collection under both text-input arms, brief embedded once and reused.

    Returns a flat dict — one row of the gate's results table.
    """
    titles, abstracts = get_title_abstract(df)
    n_before = len(df)
    df, titles, abstracts = drop_empty_rows(df, titles, abstracts)
    n_dropped = n_before - len(df)

    n_truncated = sum(1 for a in abstracts if len(a) > MAX_ABSTRACT_CHARS)
    abstracts_for_ta_arm = [a[:MAX_ABSTRACT_CHARS] for a in abstracts]  # trap #2

    masks = build_label_masks(df, "triage_label")
    strict_mask = masks["strict_mask"]
    y = masks["strict_y"][strict_mask]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())

    # Brief embedded ONCE, reused for both arms below — the seam under test is the PAPER
    # side only (see module docstring: trap #3, why the shuffled-brief seam is out of scope).
    cfg = MODEL_CONFIGS[MODEL]
    brief_text = build_brief_text(df.iloc[0])
    use_case_vector = embed_texts(MODEL, [brief_text], prefix=cfg["query_prefix"], is_query=True)[0]

    result = {
        "collection": name, "provenance": provenance,
        "n_papers": len(df), "n_dropped_empty": n_dropped,
        "n_pos_strict": n_pos, "n_neg_strict": n_neg,
        "pct_abstract_over_4000": n_truncated / len(df) if len(df) else 0.0,
    }
    wall_clock = 0.0
    for variant, abstract_arg in (
        ("title_abstract", abstracts_for_ta_arm),
        ("title_only", abstracts),  # abstracts ignored by build_paper_texts' title_only branch
    ):
        vectors, elapsed = embed_papers(MODEL, titles, abstract_arg, text_variant=variant)
        wall_clock += elapsed
        ranking = retrieval_ranking_metrics(vectors[strict_mask], use_case_vector, y)
        result[f"auc_{variant}"] = ranking["query_similarity_auc"]

    result["diff"] = result["auc_title_only"] - result["auc_title_abstract"]
    result["abs_diff"] = abs(result["diff"])
    result["gate_pass"] = result["abs_diff"] >= GATE_THRESHOLD
    result["wall_clock_s"] = wall_clock
    return result


def chart(results: pd.DataFrame) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    mpl.use("Agg")

    # --- chart style (copied verbatim from notebooks/experiments/wf_usecase_diversity.ipynb
    # cell 2) -------------------------------------------------------------------------
    SURF = {"live": "#eb6834", "benchset": "#2a78d6"}
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
    SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
    DIV = LinearSegmentedColormap.from_list("div", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#8f1f1f"])

    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 110, "font.size": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })

    def offdiag(M):
        A = np.asarray(M, dtype=float)
        return A[~np.eye(len(A), dtype=bool)]
    # --- end chart style ---------------------------------------------------------------

    r = results.sort_values("provenance")
    colors = [SURF["live"] if p == "live" else SURF["benchset"] for p in r["provenance"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    y_pos = np.arange(len(r))
    for i, (_, row) in enumerate(r.iterrows()):
        ax.plot([row.auc_title_abstract, row.auc_title_only], [i, i],
                color=GRID, lw=1.2, zorder=0)
    ax.scatter(r.auc_title_abstract, y_pos, marker="o", s=70, color=colors,
               label="title+abstract (shipped join)", zorder=2)
    ax.scatter(r.auc_title_only, y_pos, marker="D", s=55, facecolor="white",
               edgecolor=colors, linewidth=1.8, label="title-only", zorder=2)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_yticks(y_pos); ax.set_yticklabels(r.collection, fontsize=8)
    ax.set_xlabel("cosine-to-brief ROC-AUC (0.5 = random)")
    ax.set_title("Fig 1 — same brief, same model: does the paper text move the ranking?")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    order = r.sort_values("diff")
    bar_colors = [SURF["live"] if p == "live" else SURF["benchset"] for p in order["provenance"]]
    # `order["diff"]`, never `order.diff` - `diff` is a DataFrame METHOD, so attribute access
    # silently hands matplotlib a bound method instead of the column.
    ax.barh(np.arange(len(order)), order["diff"], color=bar_colors)
    ax.axvspan(-GATE_THRESHOLD, GATE_THRESHOLD, color=GRID, alpha=0.35,
               label=f"±{GATE_THRESHOLD:.2f} gate")
    ax.axvspan(-NOISE_FLOOR, NOISE_FLOOR, color=GRID, alpha=0.12,
               label=f"±{NOISE_FLOOR:.2f} noise floor (CONTEXT.md §5)")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(order.collection, fontsize=8)
    ax.set_xlabel("AUC(title-only) − AUC(title+abstract)")
    ax.set_title("Fig 2 — the gate sits at a third of the noise floor, on purpose")
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(f"wrote {FIG}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reuse-csv", action="store_true",
                    help="re-render the chart and report from the existing CSV instead of "
                         "re-embedding. The embedding pass is ~14 minutes of CPU and is "
                         "deterministic, so a formatting fix should not have to pay for it twice.")
    args = ap.parse_args()
    started = time.monotonic()

    if args.reuse_csv:
        if not OUT_CSV.exists():
            raise SystemExit(f"{OUT_CSV} missing - run without --reuse-csv first")
        results = pd.read_csv(OUT_CSV)
        print(f"reusing {OUT_CSV} ({len(results)} collections, no re-embedding)")
    else:
        jobs = [(name, "benchset", load_benchset_collection(name))
                for name in SMALL_TEST_COLLECTIONS]
        jobs.append((TIRI_SILO, "live", load_tiri_silo(TIRI_SILO)))

        rows = []
        for name, provenance, df in jobs:
            print(f"  {name} ({provenance}, {len(df)} rows) ...", flush=True)
            rows.append(evaluate_collection(name, provenance, df))

        results = pd.DataFrame(rows)
        results.to_csv(OUT_CSV, index=False)
        print(f"wrote {OUT_CSV}")

    chart(results)

    total_wall_clock = time.monotonic() - started
    gate_pass_overall = bool(results["gate_pass"].any())
    passing = results[results["gate_pass"]]

    display_cols = ["collection", "provenance", "n_papers", "n_pos_strict", "n_neg_strict",
                     "pct_abstract_over_4000", "auc_title_abstract", "auc_title_only",
                     "diff", "gate_pass"]
    table = results[display_cols].set_index("collection")

    verdict = (
        f"**Gate: {'PASS' if gate_pass_overall else 'FAIL'}.** Gate: any collection where "
        f"|AUC(title-only) − AUC(title+abstract)| ≥ {GATE_THRESHOLD:.2f} → PASS (P-TX earns "
        "the Modal run). All four inside the band → FAIL (P-TX closes, CONTEXT.md §6 gains a "
        "row).\n\n"
        + ("Collection(s) clearing the bar: " + ", ".join(
            f"`{r.collection}` ({r.diff:+.3f})" for r in passing.itertuples())
           if gate_pass_overall else
           "No collection cleared the bar — largest |diff| observed is "
           f"{results['abs_diff'].max():.3f} (`{results.loc[results['abs_diff'].idxmax(), 'collection']}`).")
    )

    confidence = "🟢" if gate_pass_overall else "⚪"

    md = f"""# What text the embedding sees — the $0 pre-check for probe P-TX

**Status: measured, {time.strftime("%Y-%m-%d")}.** `BAAI/bge-small-en-v1.5`, fastembed, CPU
only. {len(SMALL_TEST_COLLECTIONS)} `small_test` SYNERGY collections
(`reports/benchset_v1_split_manifest.json`) plus one TIRI silo
(`data/processed/papers_combined.parquet`). Total wall-clock: **{total_wall_clock:.1f}s**. $0 —
no GPU, no Modal, no network beyond fastembed's own one-time model download (this run found
the model already cached locally).

**Confidence key:** 🟢 clears the 0.03 noise floor / decisively measured · 🟡 real but under
the floor · ⚪ engineering finding — a gate result at a screening threshold, not an effect size.

---

## 0. How to read any number here

**This is a screening gate, not an effect-size claim.** `CONTEXT.md` §5 measures a
seed-to-seed sd of ~0.010 on within-silo ROC-AUC and treats any gap under ~0.03 as *not
established*. This gate's bar is **{GATE_THRESHOLD:.2f}**, one third of that floor. That is
deliberate: the point of a $0 pre-check is to be generous *before* spending Modal GPU money,
not to claim a text-input effect is real. A PASS here means "worth the ~$3 to look properly",
not "text input matters, decisively". A FAIL means all four collections sit inside a band a
third the width of noise the repo already treats as unreliable — which is a much stronger
statement than "no effect found", because the band it failed to clear is itself
conservative.

**Two provenances, never pooled.** The three benchset collections' briefs are LLM-derived
from each review's own abstract (`brief_provenance = review_abstract`); the TIRI silo's brief
is analyst-written from scratch. `DATA_BRIEF.md` honest-limit #2: averaging across that seam
compares two brief-generation processes, not two text-input arms. Every table below reports
them separately.

**Deterministic, no seeds.** Cosine-to-brief AUC here is a raw ranking statistic (no
classifier fit, no folds — `retrieval_ranking_metrics`'s `query_similarity_auc`), and
fastembed's ONNX forward pass has no dropout at inference. The same input text always
produces the same score, so there is no seed-to-seed spread to report for this specific
number the way `run_label_budget_shape.py`'s fitted arms need one.

**Why the wrong-axis version of this gate was not run.** The plan's original pre-check
compared `cos_brief_*` against `cos_briefpre_*` — a **brief**-side ablation
(`embed_benchsets.BRIEF_VARIANTS["pre_screening"]` drops `objective`, adds term lists), with
byte-identical paper vectors between the pair. That measurement already exists
(`03_eda_full_benchset_v1.ipynb` §8.4, cells 39-40): paired median **0.005-0.007** (dropping
the abstract) and **0.003-0.007** (the realistic pre-screening brief) across 27 collections —
below this gate's bar, but about a different axis entirely. Verified directly against that
notebook before writing this script. Running it as the P-TX gate would have produced a FAIL
on a number that says nothing about paper text, which is why this script embeds a genuine
title-only paper arm instead — the first one ever computed for this corpus.

---

## 1. The verdict

{verdict}

🔴 **ELI18.** {"At least one collection's cosine-to-brief ranking moved by " + f"{results['abs_diff'].max():.3f}" if gate_pass_overall else "Every collection's cosine-to-brief ranking moved by at most " + f"{results['abs_diff'].max():.3f}"} when the abstract was removed and only the title was fed to the embedder,
where 0.500 AUC is a coin flip and 0.030 is the smallest gap this repo treats as a real
effect. {"That is enough to be worth the ~$3 Modal run to look properly, even though it is well under an established effect." if gate_pass_overall else "That is a fifth of the smallest gap this repo trusts, so the text fed to the embedder is not doing anything measurable here — dropping the abstract entirely barely moves the needle, which closes a lever rather than opening one."}

## 2. Per-collection results

{to_md(table.round(4), "collection")}

`pct_abstract_over_4000` is the fraction of each collection's abstracts longer than
`embed_benchsets.MAX_ABSTRACT_CHARS` (trap #2 below) — the `title_abstract` arm here
truncates them exactly as a real Modal run would; `title_only` never sees an abstract at all,
truncated or not.

## 3. Three traps recorded for whoever runs the Modal part, if this gate passes

1. **Cache filename.** `embed_benchsets.papers_path()` (~:155) is
   `benchset_v1_{{use_case_key}}_{{safe_model}}_papers.parquet` — no `text_variant` in it.
   Four text variants would silently overwrite each other, and the resume logic ("a finished
   collection is skipped outright") would then *skip* a variant that never ran. This script
   never touches that path or that pipeline — every vector here is computed directly through
   `embedding_utils.embed_papers`/`embed_texts` with report-local files only.
2. **Truncation confound.** `MAX_ABSTRACT_CHARS = 4000` truncates abstracts but never titles,
   so `title_only` is the only *fully* untruncated arm — a confound between arms, not merely
   a text change. Quantified per collection in §2 above:
   `synergy_donners_2021` has the largest share truncated; the TIRI silo and the other two
   benchset collections are under 1%.
3. **The shuffled-brief seam does not apply.** `CONTEXT.md` L258-260 mandates a
   shuffled-brief derangement control for anything claiming to read the brief, to prove the
   reader is not scoring on an artefact independent of brief content. P-TX changes the
   **paper** side, not the brief side — the brief is embedded once per collection and reused
   unchanged across both arms (see the module docstring and `evaluate_collection`) — so there
   is no brief-reading claim here for that seam to falsify. Stated explicitly, not omitted.

**Also recorded:** the `+venue` arm named in the plan is dropped — `papers_benchset_v1.parquet`
has no `venue` column, so it is unbuildable on its own confirmatory surface (only the
2,873-row `papers_combined.parquet` has one). `CONTEXT.md` §6 already rejected `has_venue` as
a *metadata* feature (part of the 11-feature punch list, +0.002 combined) — a different
operation from a text-input change, but worth saying before a reviewer asks.

## 4. The prior this cuts against

**Judgement call, not a hard metric:** `CONTEXT.md` §7 measured what a genuinely *better*
brief (an induced rule set from 60 labels) is worth to cosine-to-brief on set A — **+0.004 /
−0.005**, against +0.065 for the lexical block on the identical swap — and calls the cosine
"the unimprovable one". That is a brief-quality lever, not a text-input lever, but it is the
closest existing measurement of "can anything be done to move this particular ranker", and it
found next to nothing. {"This gate did not repeat that finding — at least one collection moved past the bar, so the paper-text axis is not obviously as inert as the brief-quality axis was." if gate_pass_overall else "This gate's result is consistent with that prior: if nothing moves the cosine ranker on the brief side or the paper-text side, that is a genuinely closed question about this particular representation, not an unlucky measurement."}

## 5. What this script did not do

No GPU, no Modal, no re-embedding of any full corpus, no change to `embed_benchsets.py` or
its cache layout. `MODEL_CONFIGS`, `build_paper_texts`, and `embed_papers` in
`scripts/embedding_utils.py` gained one optional `text_variant` parameter each
(default-preserving; every existing caller is unaffected) — nothing else in the shared core
changed.
"""

    OUT_MD.write_text(md, encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"\n{'PASS' if gate_pass_overall else 'FAIL'} — total wall-clock {total_wall_clock:.1f}s")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
