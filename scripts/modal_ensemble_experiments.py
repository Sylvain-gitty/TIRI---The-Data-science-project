"""modal_ensemble_experiments.py — exploratory v2 improvement experiments, kept separate
from scripts/modal_ensemble_candidate.py (the shipped v1 ablation) so v1's artefacts stay
reproducible and untouched while this runs.

Deliberately does NOT import `app`/`data_vol` from modal_ensemble_candidate.py — Modal's
app object needs to be defined in the module `modal run`/`modal deploy` is pointed at, not
imported cross-module (that breaks the runner's own re-import of this file). Instead this
redeclares an app with the SAME name ("tiri-ensemble-ablation") and looks up the SAME named
volume via `modal.Volume.from_name`, which attaches to the existing volume rather than
creating a second one.

**Important, learned the hard way:** `modal deploy <file>` REPLACES an app's entire
function set with only what's defined in that file — it does NOT merge across files
targeting the same app name. Deploying a second file wiped out this file's functions mid-run
once already (a `NotFoundError` on a function that plainly existed a few minutes earlier).
Consequence: every Modal function this project's ensemble work needs — the v1 candidate
cells, every v2 experiment, and the SYNERGY validation — now lives in THIS ONE FILE. Do not
split Modal functions across multiple files again; add to this one and redeploy it.

Experiment A: does a real CatBoost hyperparameter setting beat the iterations=50/depth=4
screening setting Phase 1 used (chosen for Modal timeout reasons, not quality)? Fixed to
the winning feature set from Phase 1 (`lex + cos-brief + metadata + Jasper+Qwen3-4B concat`).

Usage:
    modal run scripts/modal_ensemble_experiments.py::hparam_sweep --seed 0
    modal run scripts/modal_ensemble_experiments.py::synergy_validation --iterations 150 --depth 4
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_NAME = "tiri-ensemble-ablation"
DATA_MOUNT = "/data"
CPU_PER_CONTAINER = 8
MEMORY_PER_CONTAINER = 8192

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
data_vol = modal.Volume.from_name("tiri-fe-data", create_if_missing=False)
app = modal.App(APP_NAME, image=image)

WINNING_VARIANT = "lex + cos-brief + metadata + Jasper+Qwen3-4B concat"
CATBOOST = "CatBoost (Ordered)"

# Generous timeout. Measured, not assumed: (iterations=200, depth=6) exceeded the full
# 3600s here, so depth costs considerably more than a naive iterations x depth estimate off
# the 263s/30-fit (iterations=50, depth=4) Phase 2 baseline would suggest - hparam_sweep now
# bumps iterations alone (150, depth held at 4) rather than depth, to stay well inside this.
EXPERIMENT_TIMEOUT = 3600


@app.function(
    volumes={DATA_MOUNT: data_vol}, timeout=EXPERIMENT_TIMEOUT,
    cpu=CPU_PER_CONTAINER, memory=MEMORY_PER_CONTAINER,
)
def run_catboost_hparam_oof(iterations: int, depth: int, seed: int) -> dict:
    """Single-seed within-silo OOF probabilities for CatBoost on the winning feature set,
    at an arbitrary (iterations, depth) - generalizes run_oof_cell (fixed at the Phase 1
    screening setting) so a hyperparameter comparison doesn't need a new app/image."""
    import pandas as pd

    from ensemble_eval_utils import within_silo_oof
    from run_ensemble_candidate import build_variants, catboost_fn

    df = pd.read_parquet(f"{DATA_MOUNT}/papers_fe.parquet")
    for col in ("year", "paper_age", "citation_count"):
        df[col] = df[col].astype("float64")

    y = df["y"].to_numpy().astype(int)
    use_case = df["use_case_key"]
    groups = df["first_author"]

    cols = build_variants(df)[WINNING_VARIANT]
    X = df[cols]
    model_fn = catboost_fn(
        cols, thread_count=CPU_PER_CONTAINER, iterations=iterations, depth=depth
    )

    oof = within_silo_oof(X, y, use_case, groups, seed, model_fn)
    return {uc: arr.tolist() for uc, arr in oof.items()}


@app.function(
    volumes={DATA_MOUNT: data_vol}, timeout=EXPERIMENT_TIMEOUT,
    cpu=CPU_PER_CONTAINER, memory=MEMORY_PER_CONTAINER,
)
def run_catboost_logo(iterations: int, depth: int) -> dict:
    """LOGO (leave-one-use-case-out) scores for CatBoost on the winning feature set, at an
    arbitrary (iterations, depth) - the defaults-selection surface CONTEXT.md §1 specifies
    ("hyperparameter search happens once, centrally, in the LOGO loop"), not within-silo
    (the production surface run_catboost_hparam_oof measures). Each LOGO fit trains on
    ~1550 pooled rows (5 of 6 use cases), not the ~250-360 rows a within-silo fold trains on
    - do not assume this is cheaper than within-silo per config; scripts/
    run_central_hparam_search.py calibrates before committing to a wide grid.
    """
    import pandas as pd

    from ensemble_eval_utils import logo
    from run_ensemble_candidate import build_variants, catboost_fn

    df = pd.read_parquet(f"{DATA_MOUNT}/papers_fe.parquet")
    for col in ("year", "paper_age", "citation_count"):
        df[col] = df[col].astype("float64")

    y = df["y"].to_numpy().astype(int)
    use_case = df["use_case_key"]

    cols = build_variants(df)[WINNING_VARIANT]
    X = df[cols]
    model_fn = catboost_fn(
        cols, thread_count=CPU_PER_CONTAINER, iterations=iterations, depth=depth
    )

    result = logo(X, y, use_case, model_fn)
    return {
        uc: {k: float(v) for k, v in row.items()}
        for uc, row in result.to_dict(orient="index").items()
    }


@app.function(
    volumes={DATA_MOUNT: data_vol}, timeout=EXPERIMENT_TIMEOUT,
    cpu=CPU_PER_CONTAINER, memory=MEMORY_PER_CONTAINER,
)
def run_catboost_axis_oof(mode: str, iterations: int, depth: int, seed: int) -> dict:
    """Single-seed within-silo OOF for the winning feature set, with a fold-safe supervised
    "relevance axis" feature (see ensemble_eval_utils.within_silo_oof_with_axis) either
    added alongside the raw embedding ("add") or standing in for it entirely ("replace") -
    the PCA/SVD-adjacent idea that's actually untested (blind PCA-64 replacing the
    embedding is already measured hurting within a silo, reports/wf_query_conditioned_findings.md
    §5; this is a supervised single direction, not unsupervised variance-preserving ones).
    """
    import pandas as pd

    from ensemble_eval_utils import within_silo_oof_with_axis
    from run_ensemble_candidate import build_variants, catboost_fn

    df = pd.read_parquet(f"{DATA_MOUNT}/papers_fe.parquet")
    for col in ("year", "paper_age", "citation_count"):
        df[col] = df[col].astype("float64")

    y = df["y"].to_numpy().astype(int)
    use_case = df["use_case_key"]
    groups = df["first_author"]

    all_cols = build_variants(df)[WINNING_VARIANT]
    emb_cols = [c for c in all_cols if c.startswith("emb_")]
    other_cols = [c for c in all_cols if not c.startswith("emb_")]

    if mode == "add":
        base_cols = other_cols + emb_cols
    elif mode == "replace":
        base_cols = other_cols
    else:
        raise ValueError(f"mode must be 'add' or 'replace', got {mode!r}")

    X_other = df[base_cols]
    X_emb = df[emb_cols]
    model_fn = catboost_fn(
        base_cols + ["emb_axis_score"], thread_count=CPU_PER_CONTAINER,
        iterations=iterations, depth=depth,
    )

    oof = within_silo_oof_with_axis(X_other, X_emb, y, use_case, groups, seed, model_fn)
    return {uc: arr.tolist() for uc, arr in oof.items()}


SYNERGY_FILENAME = "papers_fe_synergy.parquet"  # Jasper+Qwen3-4B (original)
SYNERGY_FILENAME_QWEN8B = "papers_fe_synergy_jasper_qwen8b.parquet"  # Jasper+Qwen3-8B swap


@app.function(
    volumes={DATA_MOUNT: data_vol}, timeout=EXPERIMENT_TIMEOUT,
    cpu=CPU_PER_CONTAINER, memory=MEMORY_PER_CONTAINER,
)
def run_synergy_oof(branch: str, iterations: int, depth: int, seed: int, filename: str) -> dict:
    """Validates an ensemble config's core (Tier-1b BM25 + raw embedding block) against
    SYNERGY - the only prevalence-realistic, externally-labelled surface in this repo
    (1.7-14.8% included, vs. 26-77% across TIRI's own six pools). No cos_brief (SYNERGY's
    brief embeddings aren't cached) and no metadata (not built for SYNERGY reviews) - the
    same scope limit scripts/run_synergy_recall_validation.py already documents, not a new
    one. `filename` selects which embedding pair the local feature file was built with
    (SYNERGY_FILENAME = Jasper+Qwen3-4B, SYNERGY_FILENAME_QWEN8B = Jasper+Qwen3-8B) - the
    feature columns themselves are read dynamically, so this function doesn't otherwise
    need to know which embedding model is in play.

    branch: 'catboost' or 'logreg'. No author-grouping: SYNERGY exports carry no author
    column (same note as run_synergy_recall_validation.py) - a synthetic per-row "group"
    (each row its own group) makes within_silo_oof's StratifiedGroupKFold behave exactly
    like plain StratifiedKFold, reusing the tested function unchanged.
    """
    import numpy as np
    import pandas as pd

    from ensemble_eval_utils import within_silo_oof
    from run_ensemble_candidate import catboost_fn, logreg_fn

    df = pd.read_parquet(f"{DATA_MOUNT}/{filename}")
    y = df["y"].to_numpy().astype(int)
    use_case = df["use_case_key"]
    synthetic_groups = pd.Series(np.arange(len(df)))

    feature_cols = [c for c in df.columns if c not in ("y", "use_case_key")]
    X = df[feature_cols]

    if branch == "catboost":
        model_fn = catboost_fn(
            feature_cols, thread_count=CPU_PER_CONTAINER, iterations=iterations, depth=depth
        )
    elif branch == "logreg":
        model_fn = logreg_fn(feature_cols)
    else:
        raise ValueError(f"branch must be 'catboost' or 'logreg', got {branch!r}")

    oof = within_silo_oof(X, y, use_case, synthetic_groups, seed, model_fn)
    return {uc: arr.tolist() for uc, arr in oof.items()}


def _upload_synergy_data(filename: str) -> None:
    local_path = REPO / "data" / "processed" / filename
    print(f"Uploading {local_path} ({local_path.stat().st_size / 1e6:.1f} MB) ...")
    with data_vol.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), f"/{filename}")
    print("Upload done.")


@app.local_entrypoint()
def synergy_validation(
    iterations: int = 150, depth: int = 4, seeds: int = 5,
    filename: str = SYNERGY_FILENAME, out_name: str = "wf_ensemble_v2_synergy_oof.json",
) -> None:
    import json

    _upload_synergy_data(filename)

    out_path = REPO / "reports" / out_name
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    calls = [(branch, seed) for branch in ("catboost", "logreg") for seed in range(seeds)]
    print(f"Running {len(calls)} SYNERGY OOF fetches (iterations={iterations}, depth={depth}, "
          f"filename={filename}) ...")
    for (branch, seed), res in zip(
        calls,
        run_synergy_oof.starmap(
            [(branch, iterations, depth, seed, filename) for branch, seed in calls],
            return_exceptions=True,
        ),
    ):
        if isinstance(res, Exception):
            print(f"  FAILED: {branch} seed={seed}: {res!r}")
            continue
        print(f"  done: {branch} seed={seed}")
        results.setdefault(branch, {})[str(seed)] = res
        out_path.write_text(json.dumps(results))

    print(f"\nWritten to {out_path}")


@app.local_entrypoint()
def hparam_sweep(seed: int = 0) -> None:
    """Runs the screening setting (baseline) plus one bumped setting, in parallel, and
    prints a quick before/after ROC-AUC comparison. Full report-writing happens in the
    calling notebook/script, not here - this is the compute step only.

    return_exceptions=True + a write after every config, not just at the end: a first
    attempt at a "tuned" config (iterations=200, depth=6) hit the full 3600s timeout, and
    losing the already-succeeded screening result along with it (starmap raised before the
    single end-of-run write) wasted that half of the run for nothing. Depth turned out to
    cost much more than the naive iterations x depth linear estimate in this file's
    docstring assumed - bumping iterations alone this time, depth held at the baseline's 4.
    """
    import json

    configs = [
        ("screening (iterations=50, depth=4)", 50, 4),
        ("tuned (iterations=150, depth=4)", 150, 4),
    ]
    out_path = REPO / "reports" / "wf_ensemble_v2_hparam_oof.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    print(f"Running {len(configs)} CatBoost configs on '{WINNING_VARIANT}' (seed={seed}) ...")
    for (name, it, d), res in zip(
        configs,
        run_catboost_hparam_oof.starmap(
            [(it, d, seed) for _, it, d in configs], return_exceptions=True
        ),
    ):
        if isinstance(res, Exception):
            print(f"  FAILED: {name}: {res!r}")
            continue
        print(f"  done: {name}")
        results[name] = res
        out_path.write_text(json.dumps(results))

    print(f"Written to {out_path}")
