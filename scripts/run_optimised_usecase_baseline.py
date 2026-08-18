"""What does a better-written use case do to the baseline model?

Rebuilds every brief-dependent feature from the optimised specs
(`scripts/optimise_usecases.py`) and re-runs **`notebooks/main/06_baseline_logreg.ipynb`'s
pipeline, unchanged**, on each arm — so the only thing that differs between them is *the text of
six use-case spec files*.

Why this is cheap: the papers do not change
-------------------------------------------
Optimising a use case changes the *brief*, never the corpus. So the 8,704 paper-embedding columns
in `papers_fe.parquet` are reused untouched, and the only embedding work is **6 briefs x 3 models
= 18 short strings per arm** — seconds, not a corpus pass. What is rebuilt per arm:

  `cos_brief_{jasper,qwen4b,qwen8b}` + their ranks   re-embedded brief x cached paper vectors
  the 22 `lex_*` columns                             rebuilt on CPU by `build_lexical_features`
  everything else                                    identical, by construction

Following the backbone exactly, and the three places an earlier version did not
-------------------------------------------------------------------------------
"Same pipeline" has to mean the *same* pipeline, or the difference between arms is partly the
harness. Three deviations were found by reading `04_feature_engineering.ipynb` and
`06_baseline_logreg.ipynb` side by side against the first version of this script, and all three
are fixed here:

1. **Brief text is space-joined field VALUES, with no field names.**
   `wf_embedding_model_bakeoff.ipynb` cell 6 — the notebook that wrote
   `embeddings_cache/*_usecases.parquet`, which `04_feature_engineering.ipynb` §8 then turns into
   `cos_brief_*` — builds it as `" ".join(values)` over `USE_CASE_COLS`. The first version of this
   script embedded `"use_case_name: ...\nproblem_statement: ..."`. That is a different string, so
   part of the measured gain would have been the *formatting change*, not the rewrite. `--verify`
   below re-embeds the ORIGINAL briefs and checks them against the shipped cache for exactly this.
2. **`qwen8b` is re-embedded too.** `06_baseline_logreg.ipynb` reads all three embedding blocks and
   all three `cos_brief_*` columns. Leaving `cos_brief_qwen8b` at its baseline value would feed the
   optimised model one feature computed from the OLD brief — a silent inconsistency inside the arm.
3. **`rank_cos_brief_*` uses pandas `groupby(...).rank(pct=True)`**, which is notebook 04 §8's
   average-tie convention, not an argsort.

Evaluation, and it is notebook 06's, not a new one
--------------------------------------------------
`StratifiedGroupKFold` grouped by `first_author` and stratified on `use_case x label`: outer split 0
gives `final_holdout` (~20%), inner 5-fold within the remaining pool gives the validation folds.
`StandardScaler -> PCA(50, whiten)` on the 8,704-column embedding block and median-impute -> scale
on the 33 non-embedding features, both fitted **inside** each split, then
`LogisticRegression(class_weight="balanced")`. `lex_overlap_excl_n/frac` are filled with 0 pre-split,
as notebook 06 §3 does, because there NULL genuinely means "this use case has no exclusion terms".

The design constraint that makes the number mean anything
---------------------------------------------------------
The optimised specs were built **without any access to labels, papers, or the corpus** — every
token traces to another field of the same spec file (see `optimise_usecases.py`, and
`--provenance` there prints the mapping). That matters because the cold-start features match brief
text against paper text: a brief written by reading the relevant papers would fit the brief to the
labels, and the resulting "gain" would be manufactured. `DATA_BRIEF.md` honest-limit #2 records
that exact failure in the benchset corpus.

Warnings that travel with every number below
--------------------------------------------
- **This is a POOLED model and `CONTEXT.md` §1 says no pooled model ever ships** — a LogReg reads
  `use_case_key` off the raw embedding at 96.2% accuracy, so a pooled gain can be use-case identity
  rather than brief quality. It is reported because it is the baseline that was asked for; the
  **per-use-case** rows beside it are the production-relevant view.
- Prevalence here is **26-77% positive**, roughly 20x production. F2, precision and any other
  threshold-derived number are measured in the wrong regime and do not transfer (`CONTEXT.md` §3).
- A gap under **0.03** is inside the seed-to-seed noise floor and is not established
  (`CONTEXT.md` §5).

    python scripts/run_optimised_usecase_baseline.py            # full run
    python scripts/run_optimised_usecase_baseline.py --verify   # instrument checks only, then stop
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, fbeta_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from embed_benchsets import USE_CASE_COLS  # noqa: E402  the five fields the cosine brief reads
from ensemble_eval_utils import to_md  # noqa: E402
from lexical_features import build_lexical_features  # noqa: E402
from optimise_usecases import USE_CASES, _flat  # noqa: E402

FE = REPO / "data" / "processed" / "papers_fe.parquet"
PC = REPO / "data" / "processed" / "papers_combined.parquet"
CACHE = REPO / "data" / "processed" / "embeddings_cache"
RAW = REPO / "data" / "raw"
SPEC_DIRS = {"baseline": RAW, "optimised": RAW / "optimised",
             "prose_only": RAW / "optimised_prose_only"}
OUT_MD = REPO / "reports" / "wf_optimised_usecase_baseline.md"
OUT_CSV = REPO / "reports" / "wf_optimised_usecase_baseline.csv"

# 04_feature_engineering.ipynb cell 2's EMBEDDING_MODELS, verbatim: model -> column prefix.
EMB_MODELS = {"infgrad/Jasper-Token-Compression-600M": "jasper",
              "Qwen/Qwen3-Embedding-4B": "qwen4b",
              "qwen/qwen3-embedding-8b": "qwen8b"}

# --- 06_baseline_logreg.ipynb cell 3, copied verbatim. Asserted against the file in main(). ---
TARGET_COL, POSITIVE_LABEL, NEGATIVE_LABELS = "triage_label", "positive", ["negative"]
USE_CASE_COL, GROUP_COL = "use_case_key", "first_author"
NON_EMBEDDING_FEATURE_COLS = [
    "lex_n_tokens", "lex_bm25_obj", "lex_bm25_prob", "lex_bm25_must", "lex_bm25_nice", "lex_bm25_dom",
    "lex_rank_bm25_obj", "lex_rank_bm25_prob", "lex_rank_bm25_must", "lex_rank_bm25_nice", "lex_rank_bm25_dom",
    "lex_overlap_must_n", "lex_overlap_must_frac", "lex_rank_overlap_must",
    "lex_overlap_nice_n", "lex_overlap_nice_frac", "lex_rank_overlap_nice",
    "lex_overlap_excl_n", "lex_overlap_excl_frac",
    "lex_overlap_must_per_1k", "lex_overlap_nice_per_1k", "lex_has_exclude_terms",
    "cos_brief_jasper", "rank_cos_brief_jasper",
    "cos_brief_qwen4b", "rank_cos_brief_qwen4b",
    "cos_brief_qwen8b", "rank_cos_brief_qwen8b",
    "year", "paper_age", "has_abstract", "n_authors", "citation_count",
]
PCA_N_COMPONENTS, N_SPLITS_OUTER, N_SPLITS_INNER, RANDOM_STATE = 50, 5, 5, 0
# --- end of the verbatim block ---

# Columns where NULL means "this use case declares no exclusion terms", not "unknown". Filled
# with 0 pre-split, deterministically, exactly as 06_baseline_logreg.ipynb §3 does.
EXCL_FILL_COLS = ["lex_overlap_excl_n", "lex_overlap_excl_frac"]


def safe_name(model_name: str) -> str:
    """04_feature_engineering.ipynb cell 15's cache-filename convention, verbatim."""
    return model_name.replace("/", "__").replace(":", "_")


def brief_text(fields: dict) -> str:
    """The exact string `wf_embedding_model_bakeoff.ipynb` cell 6 embeds as "the use case".

    Space-joined VALUES of the five USE_CASE_COLS, no field names, no separators. This is the
    text behind every `cos_brief_*` column in `papers_fe.parquet`, so the optimised arm has to be
    built the same way or the arms differ by formatting as well as by content.
    """
    return " ".join(str(fields[c]) for c in USE_CASE_COLS
                    if fields.get(c) is not None and str(fields[c]).strip())


def load_specs(arm: str) -> dict[str, dict]:
    d = SPEC_DIRS[arm]
    return {k: _flat(json.loads((d / f"{k}.usecase.json").read_text())) for k in USE_CASES}


def corpus_fields() -> dict[str, dict]:
    """The nine brief fields **as broadcast onto every paper row by
    `01_data_compile.ipynb`** — which is what actually built `papers_fe.parquet`, and is not
    always what the spec JSON says.

    `use_case_name` is the trap: notebook 01 takes it from its own hand-written
    `USE_CASE_REGISTRY`, not from the JSON's `name`. For `solar_leo` the JSON reads
    "solar cells for low earth orbit satellites" and the corpus reads "Solar Cells for Low Earth
    Orbit Satellites". Embedding the JSON version instead moves that brief's vector by cosine
    0.991 — five times the entire effect this script is trying to measure — so the arms are built
    from the corpus and only genuinely-rewritten fields are substituted in.
    """
    cols = list(next(iter(load_specs("baseline").values())))
    pc = pd.read_parquet(PC, columns=[USE_CASE_COL, *cols]).drop_duplicates(USE_CASE_COL)
    return {r[USE_CASE_COL]: {c: r[c] for c in cols} for _, r in pc.iterrows()}


def _same(a, b) -> bool:
    la, lb = (list(a) if isinstance(a, (list, np.ndarray)) else a,
              list(b) if isinstance(b, (list, np.ndarray)) else b)
    return la == lb


def changed_fields(arm: str) -> dict[str, list[str]]:
    """use case -> the spec fields this arm actually rewrote. Provenance, and a guard: an arm that
    changed nothing would make its whole comparison empty."""
    if arm == "baseline":
        return {uc: [] for uc in USE_CASES}
    orig, new = load_specs("baseline"), load_specs(arm)
    return {uc: [c for c in orig[uc] if not _same(new[uc][c], orig[uc][c])] for uc in USE_CASES}


def arm_fields(arm: str, verbose: bool = False, only: set[str] | None = None) -> dict[str, dict]:
    """This arm's brief fields: the corpus values, with the optimiser's edits substituted in.

    Written as a diff rather than a wholesale swap so that a field the optimiser did not touch is
    byte-identical to the one that built `papers_fe.parquet`. That makes "the only thing that
    differs between arms is the text of six specs" true by construction rather than by assertion.

    `only` restricts the substitution to a subset of fields, which is what §1c's decomposition
    uses to ask "was it the objective or the domain fields?" without writing new spec files.
    """
    corpus = corpus_fields()
    if arm == "baseline":
        return corpus
    orig, new = load_specs("baseline"), load_specs(arm)
    out = {}
    for uc, fields in corpus.items():
        f, changed = dict(fields), []
        for col in orig[uc]:
            if only is not None and col not in only:
                continue
            if not _same(new[uc][col], orig[uc][col]):
                f[col], _ = new[uc][col], changed.append(col)
        out[uc] = f
        if verbose:
            print(f"  {arm}/{uc:18s} rewritten: {', '.join(changed) or '(nothing)'}")
    return out


# Which half of `prose_only` is doing the work? `prose_only` rewrites the objective on all six
# specs and additionally fills an empty domain field on two - and those two are exactly the two
# that gain. That is either the answer or a coincidence over n=6, and it is cheap to separate.
DECOMPOSE = {"objective_only": {"objective"},
             "domain_only": {"domain_industry", "domain_application"}}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def brief_vectors(specs: dict[str, dict], arm: str) -> dict[tuple[str, str], np.ndarray]:
    """(prefix, use_case) -> brief vector. Cached on the SHA of the text that produced it, so a
    change to `brief_text` or to a spec invalidates the cache instead of silently reusing it."""
    cache_path = REPO / "data" / "processed" / f"{arm}_brief_vectors.parquet"
    texts = {k: brief_text(v) for k, v in specs.items()}
    want = {(p, k): _sha(t) for p in EMB_MODELS.values() for k, t in texts.items()}

    have: dict[tuple[str, str], np.ndarray] = {}
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        have = {(r.model, r.use_case_key): np.asarray(r.embedding, dtype=np.float64)
                for r in cached.itertuples() if want.get((r.model, r.use_case_key)) == r.text_sha}
    missing = [k for k in want if k not in have]
    if not missing:
        print(f"  {arm}: all {len(want)} brief vectors cached ({cache_path.name})")
        return have

    from embedding_utils import MODEL_CONFIGS, embed_texts
    keys = sorted(texts)
    for model, prefix in EMB_MODELS.items():
        if not any(p == prefix for p, _ in missing):
            continue
        print(f"  {arm}/{prefix}: embedding {len(keys)} briefs "
              f"(median {int(np.median([len(texts[k]) for k in keys]))} chars)")
        vecs = embed_texts(model, [texts[k] for k in keys],
                           prefix=MODEL_CONFIGS[model]["query_prefix"], is_query=True)
        for k, v in zip(keys, vecs):
            have[(prefix, k)] = np.asarray(v, dtype=np.float64)

    pd.DataFrame([{"model": p, "use_case_key": k, "text_sha": want[(p, k)],
                   "embedding": have[(p, k)].astype(np.float32)} for p, k in want]) \
        .to_parquet(cache_path, index=False)
    return have


def cached_original_vectors() -> dict[tuple[str, str], np.ndarray]:
    """The use-case vectors that actually built `papers_fe.parquet`'s `cos_brief_*` columns."""
    out = {}
    for model, prefix in EMB_MODELS.items():
        frame = pd.read_parquet(CACHE / f"{safe_name(model)}_usecases.parquet")
        for r in frame.itertuples():
            out[(prefix, r.use_case_key)] = np.asarray(r.embedding, dtype=np.float64)
    return out


def add_cos_brief(df: pd.DataFrame, vecs: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    """`04_feature_engineering.ipynb` §8, verbatim in behaviour: cosine to the use case's own
    brief vector, plus the within-use-case percentile rank via pandas `rank(pct=True)`."""
    out = df.copy()
    for prefix in EMB_MODELS.values():
        cols = [c for c in df.columns if c.startswith(f"emb_{prefix}_")]
        mat = out[cols].to_numpy(dtype=np.float64)
        brief = np.stack([vecs[(prefix, uc)] for uc in out[USE_CASE_COL]])
        denom = np.linalg.norm(mat, axis=1) * np.linalg.norm(brief, axis=1)
        col = f"cos_brief_{prefix}"
        out[col] = np.where(denom > 0, (mat * brief).sum(axis=1) / np.where(denom > 0, denom, 1),
                            np.nan)
        out[f"rank_cos_brief_{prefix}"] = out.groupby(USE_CASE_COL)[col].rank(pct=True)
    return out


def add_lexical(df: pd.DataFrame, specs: dict[str, dict], text: pd.DataFrame) -> pd.DataFrame:
    """The 22 `lex_*` columns rebuilt on this arm's briefs. `text` carries title/abstract in `df`'s
    row order; the nine flat spec fields are broadcast onto it per use case, which is exactly the
    shape `04_feature_engineering.ipynb` §5 hands to `build_lexical_features`."""
    frame = text.copy()
    for col in next(iter(specs.values())):
        frame[col] = frame[USE_CASE_COL].map(lambda k: specs[k][col])
    block = build_lexical_features(frame).add_prefix("lex_")
    out = df.copy()
    for c in block.columns:
        if c in out.columns:
            out[c] = block[c].to_numpy()
    return out


# ----------------------------------------------------------------------------- metrics


def score(y_true, proba_pos, pred) -> dict:
    """`06_baseline_logreg.ipynb` cell 30's `score`, plus precision and the two rates. Precision
    was asked for; `pred_pos_rate` is here because at 57.6% prevalence a model can buy recall by
    simply predicting positive more often, and that has to be visible beside the recall."""
    return {
        "n": int(len(y_true)), "pos_rate": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, proba_pos)),
        "avg_precision": float(average_precision_score(y_true, proba_pos)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "f1": float(fbeta_score(y_true, pred, beta=1, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "pred_pos_rate": float(np.mean(pred)),
    }


def ranking_metrics(y_true, scores_arr, fractions=(0.10, 0.20), wss_target_recall=0.95) -> dict:
    """`06_baseline_logreg.ipynb` cell 30, verbatim. Threshold-free, so unlike F2 these survive the
    prevalence problem as *relative* comparisons between arms."""
    y_true = np.asarray(y_true)
    n, n_pos = len(y_true), int(y_true.sum())
    order = np.argsort(-np.asarray(scores_arr))
    cum_pos = np.cumsum(y_true[order])
    out = {}
    for frac in fractions:
        k = max(1, int(round(frac * n)))
        out[f"recall_at_{int(frac * 100)}pct"] = float(cum_pos[k - 1] / n_pos)
    target_count = int(np.ceil(wss_target_recall * n_pos))
    screened = int(np.searchsorted(cum_pos, target_count, side="left")) + 1
    out["wss_at_95"] = float((n - screened) / n - (1 - wss_target_recall))
    return out


def bootstrap_auc_ci(y_true, scores_arr, n_boot=1000, random_state=0) -> tuple[float, float]:
    """`06_baseline_logreg.ipynb` cell 30, verbatim."""
    rng = np.random.RandomState(random_state)
    y_true, scores_arr = np.asarray(y_true), np.asarray(scores_arr)
    boot = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_true[idx], scores_arr[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def all_metrics(y, proba, pred) -> dict:
    return {**score(y, proba, pred), **ranking_metrics(y, proba)}


# ----------------------------------------------------------------------------- model


def build_model_pipeline(emb_cols: list[str]) -> Pipeline:
    """`06_baseline_logreg.ipynb` cell 30's `build_model_pipeline`, verbatim."""
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("embedding_pca", Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=PCA_N_COMPONENTS, whiten=True,
                            random_state=RANDOM_STATE))]), emb_cols),
            ("non_embedding", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())]), NON_EMBEDDING_FEATURE_COLS),
        ])),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])


def splits(df: pd.DataFrame):
    """§5's outer/inner design. Identical across arms: the split depends only on `use_case_key`,
    `y` and `first_author`, none of which any arm touches — asserted in main()."""
    strat = df[USE_CASE_COL].astype(str) + "__" + df["y"].astype(str)
    outer = list(StratifiedGroupKFold(N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_STATE)
                 .split(df, strat, groups=df[GROUP_COL]))
    pool_idx, hold_idx = outer[0]
    pool = df.iloc[pool_idx].reset_index(drop=True)
    hold = df.iloc[hold_idx].reset_index(drop=True)
    strat_in = pool[USE_CASE_COL].astype(str) + "__" + pool["y"].astype(str)
    inner = list(StratifiedGroupKFold(N_SPLITS_INNER, shuffle=True, random_state=RANDOM_STATE)
                 .split(pool, strat_in, groups=pool[GROUP_COL]))
    return pool, hold, inner


def evaluate(df: pd.DataFrame, emb_cols: list[str], arm: str) -> pd.DataFrame:
    feature_cols = emb_cols + NON_EMBEDDING_FEATURE_COLS
    pool, hold, inner = splits(df)
    rows = []

    # --- validation: the five inner folds, each scored on rows the fold's model never saw ---
    oof = np.full(len(pool), np.nan)
    fold_rows = []
    for i, (tr, va) in enumerate(inner, start=1):
        m = build_model_pipeline(emb_cols).fit(pool.iloc[tr][feature_cols], pool.iloc[tr]["y"])
        pos = list(m.classes_).index(1)
        proba = m.predict_proba(pool.iloc[va][feature_cols])[:, pos]
        oof[va] = proba
        rec = all_metrics(pool.iloc[va]["y"].to_numpy(), proba, (proba >= 0.5).astype(int))
        fold_rows.append(rec)
        rows.append({"arm": arm, "fold": f"validate/fold{i}", **rec})
    fold_df = pd.DataFrame(fold_rows)
    rows.append({"arm": arm, "fold": "validate", **fold_df.mean().to_dict()})
    rows.append({"arm": arm, "fold": "validate_sd", **fold_df.std().to_dict()})

    # --- train + test: one model refit on the whole pool, holdout touched once ---
    final = build_model_pipeline(emb_cols).fit(pool[feature_cols], pool["y"])
    pos = list(final.classes_).index(1)
    tr_p = final.predict_proba(pool[feature_cols])[:, pos]
    ho_p = final.predict_proba(hold[feature_cols])[:, pos]
    rows.append({"arm": arm, "fold": "train",
                 **all_metrics(pool["y"].to_numpy(), tr_p, (tr_p >= 0.5).astype(int))})
    lo, hi = bootstrap_auc_ci(hold["y"].to_numpy(), ho_p)
    rows.append({"arm": arm, "fold": "test", "auc_ci_low": lo, "auc_ci_high": hi,
                 **all_metrics(hold["y"].to_numpy(), ho_p, (ho_p >= 0.5).astype(int))})

    # --- per use case, on the held-out rows: the production-relevant view (CONTEXT.md §1) ---
    for uc, g in hold.assign(_s=ho_p).groupby(USE_CASE_COL):
        if g["y"].nunique() < 2:
            continue
        rows.append({"arm": arm, "fold": f"test/{uc}",
                     **all_metrics(g["y"].to_numpy(), g["_s"].to_numpy(),
                                   (g["_s"] >= 0.5).to_numpy().astype(int))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- cold start

COLD_FEATURES = ["cos_brief_jasper", "cos_brief_qwen4b", "cos_brief_qwen8b",
                 "lex_bm25_obj", "lex_bm25_must", "lex_bm25_nice", "lex_overlap_must_frac"]


def cold_start(arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The zero-label rung: each brief-reading feature scored raw, per use case, no fitting.

    This is the arm that should move, and the fitted baseline is the arm that should not.
    `wf_spec_quality_answer.md` §2 measured the regimes inverting: prose damage costs a
    zero-label matcher 0.15-0.26 ROC-AUC and a fitted one <=0.009. The baseline below is fitted on
    ~1,478 labels, i.e. **24x** the 60-label budget at which brief quality had already stopped
    mattering. Reporting only that arm would answer a different question than the one asked.

    No fitting, no threshold, no split - every row is held out from a model that does not exist.
    """
    rows = []
    for feat in COLD_FEATURES:
        for uc in sorted(arms["baseline"][USE_CASE_COL].unique()):
            rec, ok = {"feature": feat, "use_case": uc}, True
            for name, frame in arms.items():
                g = frame[frame[USE_CASE_COL] == uc]
                if g["y"].nunique() < 2 or g[feat].isna().all():
                    ok = False
                    break
                rec[name] = roc_auc_score(g["y"], g[feat].fillna(g[feat].median()))
            if ok:
                rows.append(rec)
    d = pd.DataFrame(rows)
    for name in arms:
        if name != "baseline":
            d[f"d_{name}"] = d[name] - d["baseline"]
    return d


# ----------------------------------------------------------------------------- verification


def verify(fe: pd.DataFrame, text: pd.DataFrame, embed: bool) -> dict[str, float]:
    """Can this script rebuild the SHIPPED columns from the original specs?

    Three checks, because the arm comparison is worthless if the rebuild path differs from the one
    that produced `papers_fe.parquet`:

      A  cosine + rank code, using the cached original brief vectors -> shipped `cos_brief_*`
      B  lexical code, using the original spec JSONs                 -> shipped `lex_*`
      C  brief TEXT: re-embed the original briefs and compare to the cached original vectors

    C is the one that caught the formatting bug. A and B pass with the wrong brief string too,
    because both reuse artefacts rather than rebuilding the text.

    Returns max |diff| per column, plus `B# <col>` = the number of rows that differ at all, which
    is what separates "the code changed" from "two papers scored equal and the tie broke the other
    way". Only the second happens here.
    """
    out = {}
    specs = arm_fields("baseline")

    rebuilt = add_cos_brief(fe, cached_original_vectors())
    for prefix in EMB_MODELS.values():
        for col in (f"cos_brief_{prefix}", f"rank_cos_brief_{prefix}"):
            out[f"A {col}"] = float(np.nanmax(np.abs(rebuilt[col] - fe[col])))

    lex = add_lexical(fe, specs, text)
    for col in [c for c in fe.columns if c.startswith("lex_")]:
        a, b = lex[col].astype(float), fe[col].astype(float)
        assert (a.isna() == b.isna()).all(), f"{col}: NULL pattern changed on rebuild"
        diff = (a - b).abs()
        out[f"B {col}"] = float(np.nanmax(diff)) if not diff.isna().all() else 0.0
        out[f"B# {col}"] = float((diff > 1e-9).sum())

    if embed:
        fresh = brief_vectors(specs, "baseline")
        shipped = cached_original_vectors()
        for (prefix, uc), v in fresh.items():
            s = shipped[(prefix, uc)]
            out[f"C cos(re-embed, shipped) {prefix}/{uc}"] = float(
                v @ s / (np.linalg.norm(v) * np.linalg.norm(s)))
    return out


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", action="store_true",
                    help="run the three instrument checks and stop")
    ap.add_argument("--no-verify-embed", action="store_true",
                    help="skip check C (re-embedding the ORIGINAL briefs); A and B still run")
    args = ap.parse_args()

    fe = pd.read_parquet(FE)
    missing = [c for c in NON_EMBEDDING_FEATURE_COLS if c not in fe.columns]
    assert not missing, f"schema drift vs 06_baseline_logreg.ipynb: {missing}"
    lex_cols = [c for c in fe.columns if c.startswith("lex_")]
    assert set(lex_cols) <= set(NON_EMBEDDING_FEATURE_COLS), \
        f"lex_* columns notebook 06 does not use: {sorted(set(lex_cols) - set(NON_EMBEDDING_FEATURE_COLS))}"

    # notebook 06 §2: binarise the target. §3's pre-split fill happens after verification, so the
    # rebuild is compared against the file as shipped, NULLs and all.
    fe = fe[fe[TARGET_COL].isin([POSITIVE_LABEL, *NEGATIVE_LABELS])].copy().reset_index(drop=True)
    fe["y"] = (fe[TARGET_COL] == POSITIVE_LABEL).astype(int)
    n_filled = int(fe[EXCL_FILL_COLS[0]].isna().sum())

    emb_cols = [c for c in fe.columns if c.startswith("emb_")]
    print(f"{len(fe):,} rows x {fe[USE_CASE_COL].nunique()} use cases, "
          f"{fe['y'].mean():.1%} positive")
    print(f"{len(emb_cols):,} embedding columns, {len(NON_EMBEDDING_FEATURE_COLS)} non-embedding "
          f"(notebook 06's list, verbatim)")
    print(f"lex_overlap_excl_n/frac: {n_filled} NULLs -> 0 pre-split "
          f"(no exclusion terms declared, not unknown)")

    pc = pd.read_parquet(PC, columns=["paper_id", USE_CASE_COL, "title", "abstract"])
    text = fe[["paper_id", USE_CASE_COL]].merge(pc, on=["paper_id", USE_CASE_COL], how="left",
                                                validate="one_to_one")
    assert len(text) == len(fe) and text["title"].notna().all()

    print("\n=== instrument check: can this script rebuild the SHIPPED columns? ===")
    checks = verify(fe, text, embed=not args.no_verify_embed)
    worst_a = max(v for k, v in checks.items() if k.startswith("A "))
    worst_b = max(v for k, v in checks.items() if k.startswith("B ") and not k.startswith("B# "))
    n_diff = int(sum(v for k, v in checks.items() if k.startswith("B# ")))
    print(f"A  cos_brief_* / rank_*  rebuilt from the cached original brief vectors: "
          f"max |diff| = {worst_a:.2e}")
    print(f"B  all {len(lex_cols)} lex_* rebuilt from data/raw/*.usecase.json: "
          f"max |diff| = {worst_b:.2e} on {n_diff} of {len(fe) * len(lex_cols):,} cells")
    c = {k: v for k, v in checks.items() if k.startswith("C ")}
    if c:
        print(f"C  ORIGINAL briefs re-embedded vs the shipped cache: cosine "
              f"min {min(c.values()):.6f}, mean {np.mean(list(c.values())):.6f} "
              f"(1.000000 = this script builds the same brief string notebook 04 did)")
    assert worst_a < 1e-6, f"cosine/rank rebuild does not reproduce the shipped columns: {worst_a}"
    # Every lexical VALUE column reproduces to float noise, always. `lex_rank_bm25_must` differs on
    # 0-2 of 1,848 rows depending on PYTHONHASHSEED: BM25 sums its terms in set-iteration order, so
    # two `solar_leo` papers that score equal to ~1e-15 swap places under pandas' average-tie rank.
    # Measured across four seeds: 2, 2, 0, 2. Bounded at 5 so a real regression still trips it.
    for k, v in checks.items():
        if k.startswith("B ") and "rank" not in k:
            assert v < 1e-9, f"{k}: lexical value column does not reproduce ({v:.2e})"
    assert n_diff <= 5, f"lexical rebuild differs on {n_diff} cells - too many to be rank ties"
    if c:
        assert min(c.values()) > 0.999, f"brief text mismatch: worst cosine {min(c.values()):.4f}"
    if args.verify:
        return

    fe[EXCL_FILL_COLS] = fe[EXCL_FILL_COLS].fillna(0.0)

    # --- the three arms ---
    print("\n=== building arms (papers untouched; only the six spec files differ) ===")
    arms = {"baseline": fe}
    for arm in ("optimised", "prose_only"):
        specs = arm_fields(arm, verbose=True)
        frame = add_lexical(fe, specs, text)
        frame = add_cos_brief(frame, brief_vectors(specs, arm))
        frame[EXCL_FILL_COLS] = frame[EXCL_FILL_COLS].fillna(0.0)
        arms[arm] = frame

    for arm, frame in arms.items():
        assert frame[[USE_CASE_COL, "y", GROUP_COL]].equals(fe[[USE_CASE_COL, "y", GROUP_COL]]), \
            f"{arm} moved a split-determining column"

    moved = {f"{c} [{a}]": float(np.corrcoef(fe[c].fillna(0), arms[a][c].fillna(0))[0, 1])
             for a in ("optimised", "prose_only")
             for c in ("cos_brief_jasper", "cos_brief_qwen4b", "cos_brief_qwen8b",
                       "lex_bm25_obj", "lex_bm25_must")}
    print("\ninstrument check - correlation between each arm's feature and the baseline's "
          "(1.0000 would mean the rewrite changed nothing and the comparison is empty):")
    for k, v in moved.items():
        print(f"  {k:32s} r={v:.4f}")

    # --- §1c: which half of `prose_only` is doing the work? Cold start only, because §2 shows the
    # fitted model cannot separate the arms it already has, let alone two finer ones. ---
    decomp_arms = {"baseline": fe, "prose_only": arms["prose_only"]}
    for name, cols in DECOMPOSE.items():
        spec = arm_fields("prose_only", verbose=True, only=cols)
        frame = add_lexical(fe, spec, text)
        frame = add_cos_brief(frame, brief_vectors(spec, name))
        frame[EXCL_FILL_COLS] = frame[EXCL_FILL_COLS].fillna(0.0)
        decomp_arms[name] = frame
    decomp = cold_start(decomp_arms)
    decomp.to_csv(str(OUT_CSV).replace(".csv", "_decompose.csv"), index=False)
    dc = (decomp[decomp["feature"].str.startswith("cos_brief_")]
          .groupby("use_case")[["baseline", "objective_only", "domain_only", "prose_only"]]
          .mean().round(3))
    dc["d_objective"] = (dc["objective_only"] - dc["baseline"]).round(3)
    dc["d_domain"] = (dc["domain_only"] - dc["baseline"]).round(3)
    dc["d_both"] = (dc["prose_only"] - dc["baseline"]).round(3)
    print("\n=== which half of the rewrite moved it? (cold-start cosine, mean of 3 encoders) ===")
    print(to_md(dc))

    # --- cold start ---
    cold = cold_start(arms)
    cold.to_csv(str(OUT_CSV).replace(".csv", "_coldstart.csv"), index=False)
    cs = cold.groupby("feature")[list(arms) + [f"d_{a}" for a in arms if a != "baseline"]] \
        .mean().round(4)
    for a in arms:
        if a != "baseline":
            cs[f"better_{a}"] = cold.groupby("feature")[f"d_{a}"].apply(
                lambda s: f"{int((s > 0).sum())}/{len(s)}")
    cs = cs.loc[[f for f in COLD_FEATURES if f in cs.index]]
    print("\n=== COLD START (0 labels, one feature, no fitting) - the regime that should move ===")
    print(to_md(cs))

    # --- the fitted baseline ---
    res = pd.concat([evaluate(f, emb_cols, a) for a, f in arms.items()], ignore_index=True)
    res.to_csv(OUT_CSV, index=False)

    order = (["train", "validate", "validate_sd"]
             + [f"validate/fold{i}" for i in range(1, N_SPLITS_INNER + 1)]
             + ["test"] + sorted(f for f in res["fold"].unique() if f.startswith("test/")))
    headline = ["n", "pos_rate", "f2", "roc_auc", "recall", "precision", "avg_precision", "f1",
                "pred_pos_rate", "recall_at_10pct", "recall_at_20pct", "wss_at_95"]
    main_tbl = (res.set_index(["arm", "fold"])[headline]
                .reindex(pd.MultiIndex.from_product([list(arms), order],
                                                    names=["arm", "fold"])).round(4))
    piv = res.pivot(index="fold", columns="arm")
    delta = pd.DataFrame({f"{m} [{a}]": piv[m][a] - piv[m]["baseline"]
                          for a in ("optimised", "prose_only")
                          for m in ["f2", "roc_auc", "recall", "precision"]}).loc[order].round(4)

    print("\n" + to_md(main_tbl))
    print("\noptimised - baseline / prose_only - baseline:\n" + to_md(delta))

    rewritten = pd.DataFrame({a: pd.Series({uc: ", ".join(f"`{c}`" for c in v) or "—"
                                            for uc, v in changed_fields(a).items()})
                              for a in ("optimised", "prose_only")})
    rewritten.index.name = "use case"
    OUT_MD.write_text(_report(res, main_tbl, delta, moved, fe, cs, cold, checks, rewritten, dc),
                      encoding="utf-8")
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_MD}")


def _per_use_case_cosine(cold: pd.DataFrame) -> pd.DataFrame:
    """Averaged over the three embedding models, so one use case is one row rather than three."""
    c = cold[cold["feature"].str.startswith("cos_brief_")]
    out = c.groupby("use_case")[["baseline", "prose_only", "d_prose_only"]].mean()
    out["models_better"] = c.groupby("use_case")["d_prose_only"].apply(
        lambda s: f"{int((s > 0).sum())}/3")
    out["clears_floor"] = np.where(out["d_prose_only"] >= 0.03, "🟢 yes",
                                   np.where(out["d_prose_only"] <= -0.03, "🔴 worse", "—"))
    return out.round(3).rename(columns={"baseline": "baseline_auc", "prose_only": "optimised_auc",
                                        "d_prose_only": "gain"})


def _report(res, main_tbl, delta, moved, fe, cs, cold, checks, rewritten, dc) -> str:
    c = {k: v for k, v in checks.items() if k.startswith("C ")}
    worst_a = max(v for k, v in checks.items() if k.startswith("A "))
    worst_b = max(v for k, v in checks.items() if k.startswith("B ") and not k.startswith("B# "))
    n_diff = int(sum(v for k, v in checks.items() if k.startswith("B# ")))
    test = res[res["fold"] == "test"].set_index("arm")
    pc_cos = _per_use_case_cosine(cold)
    gain_mean = float(pc_cos["gain"].mean())
    n_clear = int((pc_cos["gain"] >= 0.03).sum())
    val = res[res["fold"] == "validate"].set_index("arm")
    val_sd = res[res["fold"] == "validate_sd"].set_index("arm")
    return (
        "# Do better-written use cases move the baseline model?\n\n"
        "Status: **measured.** `scripts/optimise_usecases.py` rewrote TIRI's six specs per "
        "[`wf_spec_quality_answer.md`](wf_spec_quality_answer.md); this re-ran "
        "`notebooks/main/06_baseline_logreg.ipynb`'s pipeline — same feature list, same splits, "
        "same seeds — on each arm. The papers never change, so their 8,704 embedding columns are "
        "reused untouched and the only embedding work is 18 short brief strings per arm.\n\n"
        "🟢 clears the 0.03 noise floor · 🟡 real but under it · ⚪ engineering finding\n\n"
        "## 0. How to read any number here\n\n"
        "**The only difference between the arms is the text of six spec files.** Same papers, same "
        "paper embeddings, same splits, same seeds, same pipeline. So any difference is "
        "attributable to the brief — which is why the *provenance* of the rewrite matters more "
        "than the rewrite itself.\n\n"
        "🔴 **The rewrite never saw a label, a paper, or the corpus.** Every token in the optimised "
        "specs traces to another field of the same spec file — `performance_criteria`, "
        "`constraints`, `decision_criteria` and `notes` are all analyst-written and **read by "
        "neither** feature path. The optimisation moves the analyst's own words into the fields the "
        "model actually reads; it adds no knowledge. Had the briefs been written by reading the "
        "relevant papers, the cold-start features match brief text against paper text and the gain "
        "would be manufactured — `DATA_BRIEF.md` honest-limit #2 records that failure in the "
        "benchset corpus. Run `optimise_usecases.py --provenance` to audit the mapping.\n\n"
        "**The three arms.** `baseline` = the six specs as the analysts wrote them. `optimised` = "
        "objective enriched *and* term lists topped up. `prose_only` = objective enriched, term "
        "lists left exactly as written. The third arm exists because the two halves of the rewrite "
        "turn out to pull in opposite directions, and only a separated arm can show that.\n\n"
        "**What each metric means, and what it reads if the model does nothing.**\n\n"
        "| metric | what it measures | value if the model is useless |\n|---|---|---|\n"
        "| `roc_auc` | ranking quality: the chance a random relevant paper outranks a random "
        "irrelevant one | **0.500** (a coin flip) |\n"
        "| `f2` | the screening-appropriate accuracy score — weights recall 4x precision, because "
        "missing a relevant paper costs more than reading an irrelevant one | 0 if it never "
        f"predicts positive; ~{2 * fe['y'].mean() / (1 + fe['y'].mean()) * 1.0:.2f}–"
        f"{fe['y'].mean():.2f} if it predicts positive for everything |\n"
        "| `recall` | of the papers that really were relevant, the share the model flagged | 1.000 "
        "if it flags everything — which is why `precision` and `pred_pos_rate` sit beside it |\n"
        "| `precision` | of the papers the model flagged, the share that really were relevant | "
        f"**{fe['y'].mean():.3f}** — the prevalence, i.e. what flagging at random gets you |\n"
        "| `avg_precision` | area under the precision-recall curve; more informative than AUC when "
        f"positives are rare | **{fe['y'].mean():.3f}** (the prevalence) |\n"
        "| `recall_at_10pct` / `_20pct` | if an analyst read only the top 10% / 20% of the ranked "
        "list, what share of the relevant papers would they have found | 0.10 / 0.20 |\n"
        "| `wss_at_95` | Work Saved over Sampling at 95% recall: the extra share of the pile you "
        "can skip while still finding 95% of what matters | **0.000** |\n\n"
        "All threshold metrics are at the model's own 0.5 cutoff, nothing tuned.\n\n"
        "⚠️ **A gap under 0.03 is not established** (`CONTEXT.md` §5's seed-to-seed noise floor).\n\n"
        "⚠️ **This is a POOLED model, and `CONTEXT.md` §1 says no pooled model ever ships** — a "
        "LogReg reads `use_case_key` off the raw embedding at 96.2% accuracy, so a pooled gain can "
        "be use-case identity rather than brief quality. The per-use-case `test/*` rows are the "
        "production-relevant view.\n\n"
        f"⚠️ **Prevalence is {fe['y'].mean():.1%} positive**, roughly 20x production. Every F2, "
        "precision and threshold number here is measured in the wrong regime and does not transfer "
        "(`CONTEXT.md` §3).\n\n"
        "## 0b. ⚪ Instrument checks — is this the backbone, or a lookalike?\n\n"
        "\"Same pipeline\" has to mean the same pipeline, or part of the difference between arms is "
        "the harness. Three checks, run every time:\n\n"
        f"- **A — cosine and rank code.** Rebuilding `cos_brief_*` and `rank_cos_brief_*` from the "
        f"cached original brief vectors reproduces the shipped columns to **{worst_a:.1e}**.\n"
        f"- **B — lexical code.** Rebuilding all 22 `lex_*` columns reproduces every *value* column "
        f"exactly (to {max(v for k, v in checks.items() if k.startswith('B ') and 'rank' not in k):.1e}), "
        f"with the NULL pattern unchanged. **{n_diff} of {len(fe) * 22:,}** cells differ, all in "
        f"`lex_rank_bm25_must` — see the reproducibility note below.\n"
        + (f"- **C — the brief string itself.** Re-embedding the ORIGINAL six briefs and comparing "
           f"to the vectors that actually built `papers_fe.parquet` gives cosine "
           f"**{min(c.values()):.6f}** at worst — jasper and qwen4b return exactly 1.000000 on all "
           f"six, and the shortfall is qwen8b, which is served over an API rather than run locally."
           f" 1.000000 means this script writes the same brief text notebook 04 did.\n" if c
           else "")
        + "\n🔴 **These checks are not decoration — they caught two real bugs, and the second one "
        "was larger than the effect being measured.**\n\n"
        "1. **The brief was being formatted wrongly.** The first version of this script embedded "
        "`\"use_case_name: ...\\nproblem_statement: ...\"`, whereas "
        "`wf_embedding_model_bakeoff.ipynb` cell 6 — the notebook that wrote the cache notebook 04 "
        "reads — joins the field *values* with spaces and no field names. Part of the measured "
        "\"gain\" was the formatting change, not the rewrite.\n"
        "2. **The spec files on disk are not what the corpus carries.** `01_data_compile.ipynb` "
        "takes `use_case_name` from its own hand-written registry, not from the JSON's `name` "
        "field, and for `solar_leo` those differ in capitalisation: the JSON says *solar cells for "
        "low earth orbit satellites*, the corpus says *Solar Cells for Low Earth Orbit "
        "Satellites*. Embedding the JSON version instead moved that one brief's vector by cosine "
        "0.991 — **about five times the size of the entire effect this report measures** — and "
        "changed 2 of 1,848 `lex_rank_bm25_must` values, which is small enough to be mistaken for "
        "a rounding tie and waved through. Both arms are therefore built from the corpus values, "
        "with only genuinely-rewritten fields substituted in.\n\n"
        "⚪ **Reproducibility note, found while chasing bug 2 and worth recording on its own.** "
        "`lex_rank_bm25_must` is **not bit-reproducible across processes**. BM25 adds up one "
        "contribution per query term, and the terms come out of a Python set, whose iteration "
        "order changes with `PYTHONHASHSEED` — so the sum lands ~1e-15 apart between runs. Two "
        "`solar_leo` papers score exactly equal, and that 1e-15 decides which of them pandas' "
        "average-tie rank puts first. Measured over four seeds the affected cell count was 2, 2, "
        "0, 2 out of 1,848. It is far too small to touch any number in this report, and it is not "
        "a bug in this script — it is a property of the shipped feature pipeline, and it is the "
        "reason this check is written on *cell count* rather than on a magnitude tolerance, where "
        "a genuine two-cell regression would have hidden underneath it.\n\n"
        "**What actually changed in each spec.** Everything not listed is byte-identical to the "
        "corpus:\n\n"
        + to_md(rewritten) + "\n\n"
        "Note that `prose_only` is not *purely* the objective: where `domain_industry` or "
        "`domain_application` was empty, it is filled from the spec's own `name` or "
        "`technology_focus`. That affects two use cases and is part of the cosine brief, so it is "
        "named here rather than buried.\n\n"
        "**Did the rewrite actually change the features?** Correlation between each arm's feature "
        "and the baseline's — 1.0000 would mean the comparison is empty:\n\n"
        + "\n".join(f"- `{k}` r={v:.4f}" for k, v in moved.items()) + "\n\n"
        "## 1. Cold start — the regime this was supposed to move\n\n"
        "Each brief-reading feature scored **raw**, per use case: no labels, no fitting, no "
        "threshold, no split. Every row is held out from a model that does not exist. This is the "
        "rung a better spec is supposed to help, and it is the one the fitted baseline in §2 "
        "cannot see — §2 is fitted on ~1,478 labels, **24x** the 60-label budget at which "
        "`wf_spec_quality_answer.md` §2 found brief quality had already stopped mattering.\n\n"
        + to_md(cs) + "\n\n"
        "*ELI18: this asks \"if you had zero labelled papers and could only sort by how well each "
        "paper matches the brief, how good would that sort be?\" 0.500 is a coin flip. It is the "
        "situation every new project starts in, and the only one where the words in the brief are "
        "all the model has. `better_*` counts how many of the six use cases improved.*\n\n"
        "### 1a. 🟡 The enriched objective helps, by less than the first run said, and only for "
        "two use cases\n\n"
        f"Averaged over the three embedding models, the enriched objective is worth "
        f"**{gain_mean:+.3f}** ROC-AUC across the six use cases, and **{n_clear} of 6** clear the "
        "0.03 floor:\n\n"
        + to_md(pc_cos) + "\n\n"
        f"**Hard finding.** The gain is real but **concentrated and model-dependent**. "
        f"`carbon_capture` (+{pc_cos.loc['carbon_capture', 'gain']:.3f}) and `solar_leo` "
        f"(+{pc_cos.loc['solar_leo', 'gain']:.3f}) move; the other four sit inside the floor, and "
        f"`soil_microbiome` is marginally worse. Which embedding model you ask matters as much as "
        f"which use case: Qwen3-8B gains {cs.loc['cos_brief_qwen8b', 'd_prose_only']:+.3f} "
        f"({cs.loc['cos_brief_qwen8b', 'better_prose_only']} use cases better), Qwen3-4B "
        f"{cs.loc['cos_brief_qwen4b', 'd_prose_only']:+.3f}, and Jasper only "
        f"{cs.loc['cos_brief_jasper', 'd_prose_only']:+.3f} — Jasper actually *loses* 0.034 on "
        "`ner`. A rewrite that helps one encoder and not another is not yet a property of the "
        "brief.\n\n"
        "🔴 **This supersedes the first version of this report, which claimed +0.027 on Qwen3-4B "
        "across 4 of 6 use cases.** That run embedded the brief in a different format from the one "
        "notebook 04 uses (check C above), and roughly two-thirds of the headline gain was the "
        "formatting change rather than the rewrite. The corrected figure for that same model is "
        f"**{cs.loc['cos_brief_qwen4b', 'd_prose_only']:+.3f}, inside the noise floor.** The "
        "direction of the conclusion did not change; its size did, by a factor of about three.\n\n"
        "### 1b. ⚪ Which half of the rewrite moved it — the objective, not the domain fields\n\n"
        "`prose_only` rewrites the `objective` on all six specs and *additionally* fills an empty "
        "`domain_industry` / `domain_application` on two — and those two are exactly the two that "
        "gain. That is either the explanation or a coincidence over six use cases, so the two "
        "edits were re-run separately. No new spec files: the same diff, restricted to a subset of "
        "fields.\n\n"
        + to_md(dc) + "\n\n"
        f"**Hard finding: the objective does essentially all of the work.** Enriching the objective "
        f"alone is worth **{dc['d_objective'].mean():+.4f}** of the "
        f"**{dc['d_both'].mean():+.4f}** the full rewrite delivers; filling the domain fields is "
        f"worth **{dc['d_domain'].mean():+.4f}**, and all of that sits on one use case "
        f"(`carbon_capture` +{dc.loc['carbon_capture', 'd_domain']:.3f}). The coincidence was a "
        "coincidence.\n\n"
        f"🟡 **And on `solar_leo` the domain fill is actively slightly harmful**: the objective "
        f"alone is worth +{dc.loc['solar_leo', 'd_objective']:.3f}, both together only "
        f"+{dc.loc['solar_leo', 'd_both']:.3f}. Inside the floor, so not established — but it means "
        "there is no evidence for \"fill in every empty field\" as advice, and some against.\n\n"
        "### 1c. 🔴 The term-list half of the rewrite buys nothing and risks a lot — ship "
        "`prose_only`\n\n"
        "`prose_only` and `optimised` differ **only** in whether the analyst's term lists were "
        "topped up. Every cosine number above is identical between them — the term lists are not "
        "read by the cosine path — so these two rows are the whole story:\n\n"
        f"| | `lex_bm25_must` | `lex_overlap_must_frac` |\n|---|---|---|\n"
        f"| topping the term lists up (`optimised`) | "
        f"**{cs.loc['lex_bm25_must', 'd_optimised']:+.3f}** "
        f"({cs.loc['lex_bm25_must', 'better_optimised']} better) | "
        f"**{cs.loc['lex_overlap_must_frac', 'd_optimised']:+.3f}** "
        f"({cs.loc['lex_overlap_must_frac', 'better_optimised']}) |\n"
        f"| leaving them alone (`prose_only`) | **0.000** | **0.000** |\n\n"
        "**The two term features disagree in sign and both averages are well inside the floor, so "
        "on average the change is worth nothing.** The reason to reject it anyway is the spread "
        "underneath the average: four of six use cases do not move at all, `solar_leo` — whose "
        "must-list had only two terms — gains **+0.046**, and `ner` loses **−0.116** from a single "
        "added term. That is a coin-flip payoff attached to a large, one-sided downside. "
        "`prose_only` takes the identical cosine gain and declines the bet: **enrich the "
        "objective, and leave the analyst's term lists alone.**\n\n"
        "#### The term-list rule, v1 → v2, and where it stopped\n\n"
        "The first version of the rule padded `terms_must_include` to 8 with whatever "
        "`domain_technology_focus` and `performance_criteria[].metric` contained. It cost "
        "**−0.028** on `lex_bm25_must`. Three named fixes brought that to **−0.010**:\n\n"
        "| fix | derived from | effect |\n|---|---|---|\n"
        "| **Never promote a universal evaluation metric** (`precision`, `recall`, `f1`, `auc`…) | "
        "`ner` lost −0.118 with \"Precision\" in its must-list, matching nearly every NLP paper | "
        "`ner` no longer takes Precision/Recall/F1; domain-specific quantities like "
        "\"conversion efficiency\" are still allowed and were worth **+0.046** on `solar_leo` |\n"
        "| **Top up to a floor, never expand past it** | a spec with enough terms has nothing to "
        "gain and everything to dilute | `soil_microbiome` (10) and `tech_forecasting` (6) are now "
        "**left alone**, at exactly 0.000 |\n"
        "| **Never truncate** | capping `soil_microbiome`'s 10 hand-written terms at 8 cost −0.077 "
        "| \"5–8\" is a floor and a quality bar, **never a cap** |\n\n"
        "🟡 **One regression survives, and I am stopping rather than fixing it.** `ner` still loses "
        "**−0.116** from the single term \"News text processing\". The diagnosis is dilution rather "
        "than genericness: its tokens (`news` 0.16, `text` 0.40, `processing` 0.27 document "
        "frequency in that pool) are *not* more common than the existing ones (`entity` 0.78, "
        "`named` 0.61), but adding three moderately-common tokens to a five-token query raises "
        "scores across ~30–40% of the pool and swamps the rare discriminative ones "
        "(`disambiguation` 0.055, `linking` 0.109).\n\n"
        "🔴 **That points at a document-frequency filter on candidate terms — and I have not built "
        "it, deliberately.** This rule has already been iterated twice while watching the same "
        "six-use-case measurement. A third fix aimed at `ner` specifically would be tuning the rule "
        "on its own evaluation, which is the selection-on-holdout failure `CONTEXT.md` §4 exists to "
        "prevent. The df filter is recorded as a **pre-registered proposal** to test on the 28 "
        "benchset briefs, a surface not used for any of this — not applied here.\n\n"
        "**So `wf_spec_quality_answer.md` §4's MVP row needs one word changed and one exclusion "
        "added:** *at least* 5–8 discriminative phrases, and never promote a metric name into a "
        "term list.\n\n"
        "## 2. The fitted baseline — every fold, both arms\n\n"
        "`validate` is the mean over the five inner folds and `validate_sd` their spread; the five "
        "`validate/foldN` rows are underneath so the spread is visible rather than asserted. "
        "`train` is in-sample on the training pool and is expected to be the highest row in the "
        "table — it is a check that the model fitted, not a result. `test` is the outer holdout, "
        "touched once.\n\n"
        + to_md(main_tbl) + "\n\n"
        f"**Test-set ROC-AUC with a 95% bootstrap interval:** baseline "
        f"{test.loc['baseline', 'roc_auc']:.3f} "
        f"[{test.loc['baseline', 'auc_ci_low']:.3f}, {test.loc['baseline', 'auc_ci_high']:.3f}], "
        f"optimised {test.loc['optimised', 'roc_auc']:.3f} "
        f"[{test.loc['optimised', 'auc_ci_low']:.3f}, {test.loc['optimised', 'auc_ci_high']:.3f}], "
        f"prose_only {test.loc['prose_only', 'roc_auc']:.3f} "
        f"[{test.loc['prose_only', 'auc_ci_low']:.3f}, "
        f"{test.loc['prose_only', 'auc_ci_high']:.3f}]. *ELI18: the interval is where the score "
        "would land if you kept re-drawing the 370 holdout papers. Three intervals this wide and "
        "this overlapped cannot separate three models.*\n\n"
        "## 3. Optimised − baseline, and prose_only − baseline\n\n"
        + to_md(delta) + "\n\n"
        "## 4. The answer\n\n"
        "**How much does an improved use case improve the baseline model for the six initial use "
        "cases? On the fitted baseline, by nothing measurable — and that is the expected result, "
        "not a failed experiment.**\n\n"
        f"Every fold moves less than the fold-to-fold spread of the baseline itself. On the "
        f"held-out test set, F2 goes {test.loc['baseline', 'f2']:.3f} → "
        f"{test.loc['optimised', 'f2']:.3f} (optimised) and → {test.loc['prose_only', 'f2']:.3f} "
        f"(prose_only); ROC-AUC {test.loc['baseline', 'roc_auc']:.3f} → "
        f"{test.loc['optimised', 'roc_auc']:.3f} → {test.loc['prose_only', 'roc_auc']:.3f}. "
        f"For scale, the five validation folds of the **unchanged baseline** disagree with each "
        f"other by ±{val_sd.loc['baseline', 'f2']:.3f} on F2 and "
        f"±{val_sd.loc['baseline', 'roc_auc']:.3f} on ROC-AUC. Every delta in §3 is smaller than "
        "that, and smaller again than `CONTEXT.md` §5's 0.03 floor.\n\n"
        "**Why that is the expected result.** This model is fitted on ~1,478 labels. "
        "`wf_spec_quality_answer.md` §2 measured brief quality mattering enormously at 0 labels "
        "(0.15–0.26 ROC-AUC) and already not mattering by 60 (≤0.009). At 1,478 labels — 24x that "
        "budget — the labels have long since told the model everything the brief could have, and "
        "the brief-derived features are 4 of 8,737 columns. **A well-written brief is worth a lot "
        "at label 0 and nothing at label 1,478, and this table is the second case.** It is the "
        "wrong instrument for the question, and it is reported because it is the instrument that "
        "was asked for; §1 is the right one.\n\n"
        "**What actually moves, and by how much.** At zero labels the enriched objective is worth "
        f"**{gain_mean:+.3f}** ROC-AUC averaged over three encoders and six use cases, reaching "
        f"+{pc_cos['gain'].max():.3f} on `carbon_capture`, and it clears the noise floor on "
        f"{n_clear} of 6. Topping up the term lists is worth **nothing on average** and carries a "
        "−0.116 tail on one use case, so it should not be done.\n\n"
        "**One thing this measurement cannot tell you.** The rewrite only moved the analyst's own "
        "words between fields of the same file — deliberately, so nothing leaks. The question it "
        "answers is therefore *\"how much is currently being wasted by putting good material in "
        "fields the model does not read?\"*, and the answer for these six specs is: a little, "
        "concentrated in the two whose objectives were thinnest. It does **not** measure what a "
        "genuinely better-informed brief would be worth, because writing one requires knowledge "
        "from outside the file, and there is no leak-free way to get that here.\n"
    )


if __name__ == "__main__":
    main()
