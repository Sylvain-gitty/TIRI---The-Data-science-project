"""modal_ensemble_candidate.py — runs scripts/run_ensemble_candidate.py's feature/embedding
ablation grid on Modal instead of locally.

Why this exists: CatBoost's default `thread_count=-1` hits a severe (>100x, measured)
slowdown on this development machine (Apple Silicon macOS) — a single 5-feature, 264-row
fit never finished in 10+ minutes at the default setting; pinned to `thread_count=1` the
same fit took ~7s. This is a known CatBoost/macOS-ARM thread-oversubscription pathology,
not a real cost of the workload (264 rows and a few thousand columns is nothing for
gradient boosting on ordinary hardware). Modal's standard x86_64 containers don't have it,
and running each (feature variant, branch) cell as its own container parallelizes the
8-cell grid on top of fixing that.

Reuses scripts/run_ensemble_candidate.py's own variant/branch definitions and
scripts/ensemble_eval_utils.py's logo/within_silo harness unchanged (imported, not copied)
so the Modal run and a local run are answering the exact same question with the exact same
code — only where it executes differs.

Deploy once (or after any edit here):
    modal deploy scripts/modal_ensemble_candidate.py

Run the full ablation and write reports/wf_ensemble_v1_candidate.md (same report shape as
`python scripts/run_ensemble_candidate.py`, just executed on Modal):
    modal run scripts/modal_ensemble_candidate.py --seeds 5
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

APP_NAME = "tiri-ensemble-ablation"
REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "processed" / "papers_fe.parquet"
DATA_MOUNT = "/data"
CPU_PER_CONTAINER = 8
MEMORY_PER_CONTAINER = 8192  # MB. Ruled out as the cause of the two widest CatBoost cells'
                             # failure (confirmed genuine timeouts at 1800s even with this
                             # much memory, not an OOM) - kept anyway as cheap headroom.
CELL_TIMEOUT = 2400  # widened again after confirming (twice) that the two widest CatBoost
                     # cells were hitting a genuine per-fit time cost, not memory - paired
                     # with cutting iterations 100->50 in run_ensemble_candidate.catboost_fn.
                     # A cell hitting this again means both need revisiting once more.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas>=2.2.0", "pyarrow>=16.0.0", "numpy>=2.0.0",
        "scikit-learn>=1.4.0", "catboost>=1.2.0",
    )
    .add_local_python_source(
        "ensemble_eval_utils", "fold_pipeline_utils", "lexical_features", "run_ensemble_candidate"
    )
)

# Persistent volume so papers_fe.parquet (~100MB) uploads once per change, not once per cell.
data_vol = modal.Volume.from_name("tiri-fe-data", create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={DATA_MOUNT: data_vol}, timeout=CELL_TIMEOUT,
    cpu=CPU_PER_CONTAINER, memory=MEMORY_PER_CONTAINER,
)
def run_cell(variant_name: str, branch_name: str, seeds: int) -> dict:
    """One (feature variant, branch) cell: both evaluation surfaces, all 6 use cases."""
    import pandas as pd

    from ensemble_eval_utils import logo, within_silo
    from run_ensemble_candidate import BRANCHES, build_variants

    df = pd.read_parquet(f"{DATA_MOUNT}/papers_fe.parquet")
    for col in ("year", "paper_age", "citation_count"):
        df[col] = df[col].astype("float64")

    y = df["y"].to_numpy().astype(int)
    use_case = df["use_case_key"]
    groups = df["first_author"]

    cols = build_variants(df)[variant_name]
    X = df[cols]

    fn_factory = BRANCHES[branch_name]
    model_fn = (
        fn_factory(cols, thread_count=CPU_PER_CONTAINER)
        if branch_name == "CatBoost (Ordered)"
        else fn_factory(cols)
    )

    logo_df = logo(X, y, use_case, model_fn)
    silo_df = within_silo(X, y, use_case, groups, seeds, model_fn)
    return {
        "logo": logo_df.reset_index().to_dict(orient="records"),
        "within_silo": silo_df.reset_index().to_dict(orient="records"),
    }


@app.function(
    volumes={DATA_MOUNT: data_vol}, timeout=CELL_TIMEOUT,
    cpu=CPU_PER_CONTAINER, memory=MEMORY_PER_CONTAINER,
)
def run_oof_cell(variant_name: str, branch_name: str, seed: int) -> dict:
    """One (feature variant, branch) cell's single-seed within-silo OOF probabilities, per
    use case — what notebooks/pipelines/wf_ensemble_fold_pipeline.ipynb (Phase 2) calls for
    its diagnostic stages (calibration curves, branch-disagreement correlation, Recall@k
    curves), which need actual predictions rather than run_cell's aggregated scores.
    Same reason run_cell routes CatBoost through Modal at all: CatBoost's default
    thread_count=-1 hits a severe slowdown on the development machine's Apple Silicon (see
    this file's module docstring) — the notebook can't fit CatBoost locally either.
    """
    import pandas as pd

    from ensemble_eval_utils import within_silo_oof
    from run_ensemble_candidate import BRANCHES, build_variants

    df = pd.read_parquet(f"{DATA_MOUNT}/papers_fe.parquet")
    for col in ("year", "paper_age", "citation_count"):
        df[col] = df[col].astype("float64")

    y = df["y"].to_numpy().astype(int)
    use_case = df["use_case_key"]
    groups = df["first_author"]

    cols = build_variants(df)[variant_name]
    X = df[cols]

    fn_factory = BRANCHES[branch_name]
    model_fn = (
        fn_factory(cols, thread_count=CPU_PER_CONTAINER)
        if branch_name == "CatBoost (Ordered)"
        else fn_factory(cols)
    )

    oof = within_silo_oof(X, y, use_case, groups, seed, model_fn)
    return {uc: arr.tolist() for uc, arr in oof.items()}


def _upload_data() -> None:
    print(f"Uploading {DATA_PATH} ({DATA_PATH.stat().st_size / 1e6:.1f} MB) to the "
          f"tiri-fe-data volume ...")
    with data_vol.batch_upload(force=True) as batch:
        batch.put_file(str(DATA_PATH), "/papers_fe.parquet")
    print("Upload done.")


@app.local_entrypoint()
def main(seeds: int = 5) -> None:
    _upload_data()

    sys.path.insert(0, str(REPO / "scripts"))
    import pandas as pd
    import pyarrow.parquet as pq
    from ensemble_eval_utils import to_md
    from run_ensemble_candidate import BRANCHES, NOISE_FLOOR, OUT, PUBLISHED_BASELINE, build_variants

    # Read the variant names off the same function the remote cells use (schema only, not
    # the full ~100MB file) so this local reassembly step can never drift out of sync with
    # what actually ran remotely.
    schema_only = pd.DataFrame(columns=pq.read_schema(DATA_PATH).names)
    variant_names = list(build_variants(schema_only).keys())
    branch_names = list(BRANCHES.keys())
    cells = [(v, b) for v in variant_names for b in branch_names]

    print(f"Running {len(cells)} cells on Modal ({CPU_PER_CONTAINER} CPUs/container) ...")
    # return_exceptions=True: one cell timing out must not erase visibility into the other
    # 7 that succeeded - print each result as it lands rather than waiting on list() so a
    # long tail is visible while it's still running, not just at the end.
    logo_results = {}
    silo_results = {}
    failed_cells: list[str] = []
    for (vname, bname), res in zip(
        cells, run_cell.starmap([(v, b, seeds) for v, b in cells], return_exceptions=True)
    ):
        key = f"{vname} [{bname}]"
        if isinstance(res, Exception):
            print(f"  FAILED: {key}: {res!r}")
            failed_cells.append(key)
            continue
        print(f"  done: {key}")
        logo_results[key] = pd.DataFrame(res["logo"]).set_index("use_case")
        silo_results[key] = pd.DataFrame(res["within_silo"]).set_index("use_case")

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("# Ensemble v1 candidate — CatBoost vs LogisticRegression, feature/embedding ablation")
    emit()
    emit(f"Run on Modal ({len(cells)} cells, {CPU_PER_CONTAINER} CPUs/container, "
         f"{MEMORY_PER_CONTAINER}MB) rather than locally — CatBoost's default "
         f"`thread_count=-1` hits a severe slowdown on this machine's Apple Silicon (a "
         f"single tiny fit never finished in 10+ minutes; `thread_count=1` took ~7s for the "
         f"same fit). CatBoost also runs at `iterations=50, depth=4` (not CatBoost's own "
         f"1000/6 defaults) — measured directly, the widest variant (~4600 raw embedding "
         f"columns, 150 sequential fits per cell) twice exceeded a {CELL_TIMEOUT}s per-cell "
         f"timeout at 100 iterations even with 8GB of memory (a genuine per-fit time cost, "
         f"not memory), and this is a screening pass across 8 cells, not the final tuned "
         f"model. Otherwise identical ablation to `python scripts/run_ensemble_candidate.py`.")
    emit()
    n_rows = pq.ParquetFile(DATA_PATH).metadata.num_rows
    emit(f"`{n_rows}` labelled papers, 6 use cases. Gate: a "
         f"combination only carries into Phase 2 if its within-silo ROC-AUC beats the "
         f"published Tier1b+embedding LogisticRegression baseline "
         f"(`reports/wf_tier1b_lexical_control.md` §2) by more than the "
         f"~{NOISE_FLOOR:.2f} noise floor (CONTEXT.md §5) on a majority of use cases.")
    emit()
    if failed_cells:
        emit(f"**{len(failed_cells)}/{len(cells)} cells did not complete** (hit the "
             f"{CELL_TIMEOUT}s per-cell timeout or errored) and are excluded below, not "
             f"silently padded: {', '.join(failed_cells)}.")
        emit()

    emit("## 1. Leave-one-use-case-out (chooses central defaults, not a production number)")
    emit()
    logo_summary = pd.DataFrame({
        name: r[["roc_auc", "pr_auc", "recall_at_10pct"]].mean() for name, r in logo_results.items()
    }).T.round(3)
    emit(to_md(logo_summary, "variant [branch]"))
    emit()

    emit("## 2. Within-silo, author-grouped 5-fold (the production shape, and the gate)")
    emit()
    silo_summary = pd.DataFrame({
        name: r[["roc_auc", "pr_auc", "recall_at_10pct"]].mean() for name, r in silo_results.items()
    }).T.round(3)
    emit(to_md(silo_summary, "variant [branch]"))
    emit()

    emit("Per use case, within-silo ROC-AUC:")
    emit()
    per_uc_raw = pd.DataFrame({name: r["roc_auc"] for name, r in silo_results.items()})
    per_uc_csv = REPO / "reports" / "wf_ensemble_v1_candidate_within_silo.csv"
    per_uc_raw.to_csv(per_uc_csv)
    per_uc = per_uc_raw.round(3)
    emit(to_md(per_uc, "use_case"))
    emit()

    emit("## 3. Gate: delta vs. published Tier1b+embedding LogisticRegression baseline")
    emit()
    baseline = pd.Series(PUBLISHED_BASELINE)
    delta = per_uc.sub(baseline, axis=0).round(3)
    emit(to_md(delta, "use_case (delta ROC-AUC)"))
    emit()
    for name in per_uc.columns:
        clears = int((delta[name] > NOISE_FLOOR).sum())
        emit(f"- **{name}**: clears the noise floor on {clears}/6 use cases "
             f"(mean delta {delta[name].mean():+.3f}).")
    emit()

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWritten to {OUT.relative_to(REPO)}")
