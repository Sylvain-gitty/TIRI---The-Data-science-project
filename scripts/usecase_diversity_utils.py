"""usecase_diversity_utils.py — one loader + one metric set for the use-case
diversity / overlap map, across BOTH the TIRI live corpus and benchset_v1.

WHY THIS EXISTS
---------------
`notebooks/experiments/wf_usecase_diversity.ipynb` asks a question no existing script
asks: not "how well does a representation separate positives from negatives inside a use
case" (that is compare_embeddings.py / latent_space_utils.py) but "how far apart are the
USE CASES THEMSELVES, and does that distance depend on which embedding model you ask".

Two things make that awkward enough to deserve a module rather than notebook cells:

1. **Two cache layouts.** The live corpus caches one `<safe_model>_papers.parquet` for all
   1,852 labelled papers keyed by `paper_id`; benchset_v1 caches one file per collection
   (`benchset_v1_<use_case_key>_<safe_model>_papers.parquet`) because `paper_id` is only
   unique *within* a collection there — see embed_benchsets.py's docstring. Every caller
   wants "give me use case X's vectors", not "remember which layout X lives in".
2. **2.4 GB of cached vectors.** Reading all of it per notebook re-run is wasteful when
   every question here is answered by (a) an exact per-use-case centroid and (b) a seeded
   subsample. `build_diversity_cache` does the full pass ONCE and writes ~34 centroids
   plus `SUBSAMPLE_PER_USE_CASE` vectors each; the notebook reads only that.

WHAT IS AND IS NOT COMPARABLE ACROSS THE TWO SURFACES
-----------------------------------------------------
Comparable, and this is the whole reason a joint map is legitimate:

- **Briefs.** embed_benchsets.py builds benchset brief text from the same five columns in
  the same order with the same join as wf_embedding_model_bakeoff.ipynb used for the live
  briefs (`use_case_name problem_statement objective domain_industry domain_application`),
  and both go through `embed_texts(..., prefix=query_prefix, is_query=True)`. Verified
  field-for-field, not assumed — `embed_benchsets.BRIEF_VARIANTS["full"] == USE_CASE_COLS`
  is asserted in `assert_surfaces_comparable()` below.
- **Papers.** Both sides route through `embedding_utils.embed_papers`, so the title/
  abstract join and passage prefix are identical.

Two documented asymmetries, carried as caveats rather than silently smoothed over:

- benchset abstracts are truncated to 4,000 characters (embed_benchsets.py's GPU-OOM fix);
  the live corpus was embedded without that cap. It touches 0.2% of benchset rows.
- benchset briefs are **AI-written** from each review's title+abstract and were never
  written by the customer (`brief_provenance`); live briefs are analyst-written. A
  brief-space distance between a live and a benchset use case therefore mixes a topic
  difference with a provenance difference. Corpus-space distances do not have this problem
  — that is why both spaces are reported side by side and never averaged together.

Only `jasper` and `qwen4b` are cached on both surfaces. The live surface has 11 models,
which is what makes the "is this structure real or is it the model's opinion" check in §5
of the notebook possible at all; that check is live-only by necessity.

THE ANISOTROPY TRAP
-------------------
Raw cosine between two use-case centroids is NOT "how similar these use cases are". Every
sentence embedding of English academic prose shares a large common direction, so two
unrelated corpora routinely sit at cosine 0.7-0.9. Two guards, both used by the notebook:

- `centroid_similarity(..., center=True)` subtracts the pooled mean vector first, which
  removes the shared component and leaves the part that actually distinguishes use cases.
  (This is a *diagnostic* re-centring of the similarity map. It is NOT the per-use-case
  mean-centring CONTEXT.md §6 rejected as a feature transform — different operation,
  different purpose, and nothing here feeds a classifier.)
- `split_half_reference` gives the only honest yardstick: split ONE use case's papers in
  two and measure the same quantity. That is what "as similar as it is possible to be"
  scores on this metric with this model. Read every off-diagonal against it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_benchsets import BRIEF_VARIANTS, USE_CASE_COLS  # noqa: E402
from latent_space_utils import normalize_rows  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BENCHSETS = REPO / "data" / "benchsets_v1"
CACHE = REPO / "data" / "processed" / "embeddings_cache"
DIVERSITY_CACHE = REPO / "data" / "processed" / "usecase_diversity_cache"

LIVE, BENCH = "live", "benchset"

# short key -> HuggingFace/API model name, exactly as the cache filenames encode it.
# `surfaces` records where vectors actually exist; only jasper/qwen4b span both.
MODELS: dict[str, dict] = {
    "jasper":    {"name": "infgrad/Jasper-Token-Compression-600M", "dim": 2048, "surfaces": (LIVE, BENCH)},
    "qwen4b":    {"name": "Qwen/Qwen3-Embedding-4B",               "dim": 2560, "surfaces": (LIVE, BENCH)},
    "qwen8b":    {"name": "qwen/qwen3-embedding-8b",               "dim": 4096, "surfaces": (LIVE,)},
    "openai3l":  {"name": "openai/text-embedding-3-large",         "dim": 3072, "surfaces": (LIVE,)},
    "gemini2":   {"name": "google/gemini-embedding-2",             "dim": 3072, "surfaces": (LIVE,)},
    "qzhou":     {"name": "Kingsoft-LLM/QZhou-Embedding",          "dim": 3584, "surfaces": (LIVE,)},
    "nemotron":  {"name": "nvidia/nemotron-3-embed-1b_free",       "dim": 2048, "surfaces": (LIVE,)},
    "pplx":      {"name": "perplexity/pplx-embed-v1-4b",           "dim": 2560, "surfaces": (LIVE,)},
    "mistral":   {"name": "mistralai/mistral-embed-2312",          "dim": 1024, "surfaces": (LIVE,)},
    "bge_m3":    {"name": "baai/bge-m3",                           "dim": 1024, "surfaces": (LIVE,)},
    "specter2":  {"name": "allenai/specter2",                      "dim":  768, "surfaces": (LIVE,)},
}
BOTH_SURFACE_MODELS = [k for k, v in MODELS.items() if len(v["surfaces"]) == 2]

# Enough to pin a centroid and a k-NN neighbourhood without holding 181k vectors in RAM.
# 500 x 34 use cases x 4,608 dims (both models) is ~310 MB in float32; the smallest
# collection (roadfreight_metareview, 123 rows) is taken whole.
SUBSAMPLE_PER_USE_CASE = 500
SEED = 42


def safe_name(model_name: str) -> str:
    """Cache-filename encoding, matching wf_embedding_model_bakeoff.ipynb's `_safe_name`."""
    return model_name.replace("/", "__").replace(":", "_")


def assert_surfaces_comparable() -> None:
    """Fail loudly if the two surfaces ever stop sharing a brief recipe.

    The joint live+benchset map is only meaningful because both sides embed the same five
    fields joined the same way. That is a fact about two files that can drift apart, so it
    is checked at import-into-notebook time rather than trusted."""
    expected = ["use_case_name", "problem_statement", "objective",
                "domain_industry", "domain_application"]
    assert set(USE_CASE_COLS) == set(expected), (
        f"embed_benchsets.USE_CASE_COLS changed to {USE_CASE_COLS}; the live briefs in the "
        "cache were built from {expected} and the two surfaces are no longer comparable."
    )
    assert BRIEF_VARIANTS["full"] == USE_CASE_COLS, (
        "the benchset 'full' brief variant no longer equals the live brief recipe"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The use-case index — one row per use case, both surfaces
# ─────────────────────────────────────────────────────────────────────────────

def use_case_index() -> pd.DataFrame:
    """use_case_key, use_case_name, surface, n_papers, n_pos, prevalence — for all 34.

    Labels come from the surface's own source of truth: papers_fe.parquet for live (the
    deduplicated labelled table the models are actually fitted on) and the per-collection
    parquet in data/benchsets_v1/ for benchset. `pass` and unlabelled live rows are already
    excluded from papers_fe, which is the right population here: it is the one the ensemble
    ever sees."""
    rows = []

    fe = pq.read_table(REPO / "data/processed/papers_fe.parquet",
                       columns=["paper_id", "use_case_key", "y"]).to_pandas()
    names = pq.read_table(REPO / "data/processed/papers_combined.parquet",
                          columns=["use_case_key", "use_case_name"]).to_pandas().drop_duplicates()
    name_map = dict(zip(names.use_case_key, names.use_case_name))
    for uc, grp in fe.groupby("use_case_key", observed=True):
        rows.append(dict(use_case_key=uc, use_case_name=name_map.get(uc, uc), surface=LIVE,
                         n_papers=len(grp), n_pos=int(grp.y.sum())))

    briefs = pd.read_parquet(BENCHSETS / "briefs.parquet")
    bench_names = dict(zip(briefs.use_case_key, briefs.use_case_name))
    for path in sorted(BENCHSETS.glob("*.parquet")):
        if path.stem in {"briefs"}:
            continue
        lab = pq.read_table(path, columns=["triage_label"]).to_pandas()
        rows.append(dict(use_case_key=path.stem, use_case_name=bench_names.get(path.stem, path.stem),
                         surface=BENCH, n_papers=len(lab),
                         n_pos=int((lab.triage_label == "positive").sum())))

    idx = pd.DataFrame(rows)
    idx["prevalence"] = idx.n_pos / idx.n_papers
    return idx.sort_values(["surface", "use_case_key"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Vector loading — the two cache layouts, behind one call
# ─────────────────────────────────────────────────────────────────────────────

def _read_embedding_parquet(path: Path) -> tuple[list[str], np.ndarray]:
    """paper_ids + an (n, dim) float32 matrix, without going through a pandas object
    column. `ListArray.flatten()` + reshape keeps the 48k-row collections at seconds and
    a few hundred MB instead of a Python float per dimension."""
    table = pq.read_table(path)
    ids = table.column("paper_id").to_pylist()
    col = table.column("embedding").combine_chunks()
    flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    return ids, flat.reshape(len(ids), -1)


def load_paper_vectors(model_key: str, use_case_key: str, surface: str) -> tuple[list[str], np.ndarray]:
    """One use case's paper vectors, whichever cache layout it lives in."""
    stem = safe_name(MODELS[model_key]["name"])
    if surface == BENCH:
        path = CACHE / f"benchset_v1_{use_case_key}_{stem}_papers.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{model_key}/{use_case_key}: no cache at {path}")
        return _read_embedding_parquet(path)

    path = CACHE / f"{stem}_papers.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{model_key}: no live cache at {path}")
    ids, V = _read_embedding_parquet(path)
    # Live paper_id is "<use_case_key>__<source_id>" (see data/processed/README.md §1),
    # so the pooled file splits by prefix without a join.
    keep = np.array([pid.rsplit("__", 1)[0] == use_case_key for pid in ids])
    return [pid for pid, k in zip(ids, keep) if k], V[keep]


def load_brief_vectors(model_key: str, surface: str, variant: str = "full") -> dict[str, np.ndarray]:
    """use_case_key -> brief vector. `variant` applies to benchset only (the live cache
    predates variants and holds exactly the `full` recipe)."""
    stem = safe_name(MODELS[model_key]["name"])
    if surface == BENCH:
        frame = pd.read_parquet(CACHE / f"benchset_v1_{stem}_usecases.parquet")
        frame = frame[frame["variant"] == variant]
    else:
        frame = pd.read_parquet(CACHE / f"{stem}_usecases.parquet")
    return {k: np.asarray(v, dtype=np.float32)
            for k, v in zip(frame["use_case_key"], frame["embedding"])}


def load_labels(use_case_key: str, surface: str) -> pd.Series:
    """paper_id -> y (1 = positive), for the population the index counts."""
    if surface == BENCH:
        lab = pq.read_table(BENCHSETS / f"{use_case_key}.parquet",
                            columns=["paper_id", "triage_label"]).to_pandas()
        return pd.Series((lab.triage_label == "positive").astype(int).to_numpy(),
                         index=lab.paper_id)
    fe = pq.read_table(REPO / "data/processed/papers_fe.parquet",
                       columns=["paper_id", "use_case_key", "y"]).to_pandas()
    fe = fe[fe.use_case_key == use_case_key]
    return pd.Series(fe.y.to_numpy(), index=fe.paper_id)


# ─────────────────────────────────────────────────────────────────────────────
# The compact cache the notebook actually reads
# ─────────────────────────────────────────────────────────────────────────────

def build_diversity_cache(model_keys: list[str] | None = None, force: bool = False) -> pd.DataFrame:
    """One full pass over the 2.4 GB vector cache; writes per-model compact artefacts.

    Per (model, use case) it stores:
      - `centroid`      exact, over EVERY paper in the use case (not the subsample)
      - `centroid_pos`  exact, over the positives only — the two can disagree, and the
                        positive centroid is the one a relevance model chases
      - a seeded subsample of `SUBSAMPLE_PER_USE_CASE` paper vectors, for the k-NN /
        classifier / split-half work that needs a distribution rather than a mean

    Returns a summary frame. Re-running is a no-op unless `force`."""
    DIVERSITY_CACHE.mkdir(parents=True, exist_ok=True)
    index = use_case_index()
    summary = []

    for model_key in (model_keys or list(MODELS)):
        cent_path = DIVERSITY_CACHE / f"{model_key}_centroids.parquet"
        samp_path = DIVERSITY_CACHE / f"{model_key}_subsample.parquet"
        if cent_path.exists() and samp_path.exists() and not force:
            summary.append(dict(model=model_key, status="cached",
                                n_use_cases=len(pd.read_parquet(cent_path))))
            continue

        rng = np.random.default_rng(SEED)
        cent_rows, samp_rows = [], []
        for _, meta in index.iterrows():
            if meta.surface not in MODELS[model_key]["surfaces"]:
                continue
            ids, V = load_paper_vectors(model_key, meta.use_case_key, meta.surface)
            y = load_labels(meta.use_case_key, meta.surface).reindex(ids)
            # A cached vector with no label is a row the labelled population dropped
            # (live `pass`/unreviewed, benchset duplicate removal) — out of scope here.
            keep = y.notna().to_numpy()
            ids = [i for i, k in zip(ids, keep) if k]
            V, y = normalize_rows(V[keep]), y[keep].to_numpy().astype(int)

            centroid = V.mean(axis=0)
            centroid_pos = V[y == 1].mean(axis=0) if (y == 1).sum() >= 2 else np.full(V.shape[1], np.nan, np.float32)
            cent_rows.append(dict(use_case_key=meta.use_case_key, surface=meta.surface,
                                  n_papers=len(V), n_pos=int(y.sum()),
                                  centroid=centroid.astype(np.float32),
                                  centroid_pos=centroid_pos.astype(np.float32)))

            take = rng.permutation(len(V))[:SUBSAMPLE_PER_USE_CASE]
            samp_rows.append(pd.DataFrame({
                "use_case_key": meta.use_case_key, "surface": meta.surface,
                "paper_id": [ids[i] for i in take], "y": y[take],
                "embedding": list(V[take]),
            }))

        pd.DataFrame(cent_rows).to_parquet(cent_path, index=False)
        pd.concat(samp_rows, ignore_index=True).to_parquet(samp_path, index=False)
        summary.append(dict(model=model_key, status="built", n_use_cases=len(cent_rows)))

    return pd.DataFrame(summary)


def load_centroids(model_key: str) -> pd.DataFrame:
    return pd.read_parquet(DIVERSITY_CACHE / f"{model_key}_centroids.parquet")


def load_subsample(model_key: str) -> pd.DataFrame:
    return pd.read_parquet(DIVERSITY_CACHE / f"{model_key}_subsample.parquet")


def stack(frame: pd.DataFrame, col: str = "embedding") -> np.ndarray:
    return np.vstack([np.asarray(v, dtype=np.float32) for v in frame[col]])


# ─────────────────────────────────────────────────────────────────────────────
# Similarity, and the yardsticks that make it readable
# ─────────────────────────────────────────────────────────────────────────────

def cosine_matrix(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    A = normalize_rows(np.asarray(A, dtype=np.float64))
    B = A if B is None else normalize_rows(np.asarray(B, dtype=np.float64))
    return A @ B.T


def centroid_similarity(centroids: pd.DataFrame, col: str = "centroid",
                        center: bool = False) -> pd.DataFrame:
    """Use-case x use-case cosine between centroids.

    `center=True` subtracts the mean centroid first — see this module's docstring on the
    anisotropy trap. Read the centred version for *structure* (who is near whom) and the
    raw version only alongside `split_half_reference`."""
    M = stack(centroids, col)
    keep = ~np.isnan(M).any(axis=1)
    M, keys = M[keep], centroids.use_case_key.to_numpy()[keep]
    if center:
        M = M - M.mean(axis=0, keepdims=True)
    return pd.DataFrame(cosine_matrix(M), index=keys, columns=keys)


def split_half_reference(model_key: str, n_splits: int = 20, center_vec: np.ndarray | None = None,
                         seed: int = SEED) -> pd.DataFrame:
    """The ceiling: split ONE use case's papers in half and score the two halves against
    each other with the same centroid-cosine used off-diagonal.

    Without this, "these two use cases sit at cosine 0.78" is uninterpretable — 0.78 could
    be near-identical or barely related depending on the model's anisotropy and the
    within-use-case spread. This is the number that says which."""
    samp = load_subsample(model_key)
    rng = np.random.default_rng(seed)
    out = []
    for (uc, surface), grp in samp.groupby(["use_case_key", "surface"], observed=True):
        V = stack(grp)
        if len(V) < 20:
            continue
        sims = []
        for _ in range(n_splits):
            perm = rng.permutation(len(V))
            a, b = V[perm[: len(V) // 2]], V[perm[len(V) // 2:]]
            ca, cb = a.mean(axis=0), b.mean(axis=0)
            if center_vec is not None:
                ca, cb = ca - center_vec, cb - center_vec
            sims.append(float(cosine_matrix(ca[None, :], cb[None, :])[0, 0]))
        out.append(dict(use_case_key=uc, surface=surface, n=len(V),
                        split_half_cos=float(np.mean(sims)), sd=float(np.std(sims))))
    return pd.DataFrame(out)


def brief_transfer_auc(model_key: str, variant: str = "full") -> pd.DataFrame:
    """The functional overlap matrix: rank use case B's papers by cosine to use case A's
    brief, and score the resulting ranking against B's OWN labels (ROC-AUC).

    Rows = the brief doing the ranking, columns = the corpus being ranked. The diagonal is
    the honest zero-label ranker CONTEXT.md §2 recommends below ~25 labels. An off-diagonal
    cell close to its column's diagonal means someone else's brief screens this use case
    nearly as well as its own — which is *functional* overlap, and is what actually
    threatens a LOGO-selected default. Two use cases can sit far apart in centroid space
    and still overlap here, so this is not a restatement of the centroid matrix.

    Deliberately the same shape of test as the shuffled-brief control in
    scripts/run_tier1b_control.py (CONTEXT.md §7 calls that the pattern to copy): score a
    real feature against a deliberately wrong brief and see what survives.

    Runs over EVERY paper, not the subsample: at benchset prevalence (0.17% in the worst
    collection) a 500-row random subsample frequently catches no positive at all, and an
    AUC on 1-2 positives is noise. A full pass is only a matrix product per collection."""
    from sklearn.metrics import roc_auc_score

    index = use_case_index()
    index = index[index.surface.isin(MODELS[model_key]["surfaces"])]

    briefs: dict[str, np.ndarray] = {}
    for surface in MODELS[model_key]["surfaces"]:
        briefs.update(load_brief_vectors(model_key, surface, variant=variant))

    keys = [k for k in index.use_case_key if k in briefs]
    B = normalize_rows(np.vstack([briefs[k] for k in keys]).astype(np.float32))

    rows = []
    for uc in keys:
        surface = index.loc[index.use_case_key == uc, "surface"].iloc[0]
        ids, V = load_paper_vectors(model_key, uc, surface)
        y = load_labels(uc, surface).reindex(ids)
        keep = y.notna().to_numpy()
        V, y = normalize_rows(V[keep]), y[keep].to_numpy().astype(int)
        if len(np.unique(y)) < 2:
            continue
        S = V @ B.T  # (n_papers, n_briefs)
        rows.append(pd.Series({src: roc_auc_score(y, S[:, j]) for j, src in enumerate(keys)},
                              name=uc))
    # index = brief doing the ranking, columns = corpus being ranked
    return pd.DataFrame(rows).T


def neighbour_leakage(model_key: str, k: int = 25) -> pd.DataFrame:
    """For each use case, the share of each paper's k nearest neighbours (in the pooled
    subsample of every use case) that come from a DIFFERENT use case, plus which use case
    takes the largest share of them.

    Centroid distance is a summary of two clouds; this is about where the clouds actually
    touch. A use case with 0% foreign neighbours is an island regardless of how its mean
    scores; one with 40% shares a real neighbourhood with someone."""
    samp = load_subsample(model_key).reset_index(drop=True)
    V = normalize_rows(stack(samp).astype(np.float32))
    owner = samp.use_case_key.to_numpy()

    # Blocked, because the full 17,000 x 17,000 similarity matrix is 1.2 GB in float32 and
    # only the top-k of each row is ever read.
    nn = np.empty((len(V), k), dtype=np.int32)
    for start in range(0, len(V), 2000):
        stop = min(start + 2000, len(V))
        S = V[start:stop] @ V.T
        S[np.arange(stop - start), np.arange(start, stop)] = -np.inf  # exclude self
        nn[start:stop] = np.argpartition(-S, kth=k, axis=1)[:, :k]
    foreign = owner[nn] != owner[:, None]

    rows = []
    for uc in samp.use_case_key.unique():
        m = owner == uc
        neigh = owner[nn[m]].ravel()
        others = pd.Series(neigh[neigh != uc]).value_counts()
        rows.append(dict(
            use_case_key=uc,
            surface=samp.surface[m].iloc[0],
            foreign_frac=float(foreign[m].mean()),
            top_neighbour=others.index[0] if len(others) else None,
            top_neighbour_frac=float(others.iloc[0] / (m.sum() * k)) if len(others) else 0.0,
        ))
    return pd.DataFrame(rows).sort_values("foreign_frac", ascending=False).reset_index(drop=True)


def effective_n_use_cases(sim: pd.DataFrame) -> float:
    """How many *independent* use cases a similarity matrix is worth.

    Participation ratio of the eigenvalue spectrum — (sum L)^2 / sum(L^2) — the same
    statistic latent_space_utils.dispersion_metrics already uses for effective
    dimensionality, applied to the use-case Gram matrix instead. 28 use cases that are
    perfectly independent score 28; 28 copies of one use case score 1. This is the number
    that says how much evidence a 28-fold LOGO mean really rests on."""
    eig = np.linalg.eigvalsh(np.asarray(sim, dtype=np.float64))
    eig = np.clip(eig, 0, None)
    return float(eig.sum() ** 2 / (eig ** 2).sum())
