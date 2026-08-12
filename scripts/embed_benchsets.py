"""embed_benchsets.py — Jasper + Qwen3-4B vectors for data/benchsets_v1, resumably.

WHY THIS EXISTS AS A SCRIPT AND NOT A NOTEBOOK CELL
---------------------------------------------------
The two embedding entry points in embedding_utils.py were built for a 2,873-paper corpus
and neither survives 181,199 papers unattended:

- `embed_via_modal` sends the whole list in ONE `.remote()` call, under the Modal app's
  3600 s timeout. 48,343 texts (synergy_walker_2018) in one call is a timeout, and the
  return payload would be ~1 GB of Python floats.
- `embed_via_openrouter` batches 64 at a time but has NO resume — one 5xx after the
  retry budget and hours of paid work are gone.

So this script adds the one thing missing, durability, and nothing else. Every actual
embedding call still goes through `embedding_utils.embed_papers` / `embed_texts`, so the
per-model title/abstract join, prefixes and query/passage asymmetry stay defined in
exactly one place (CLAUDE.md's shared-core rule). If a model's handling needs to change,
it changes there, not here.

MODEL CHOICE
------------
`infgrad/Jasper-Token-Compression-600M` (2048-dim) + `Qwen/Qwen3-Embedding-4B` (2560-dim)
— the pairing reports/wf_embedding_bakeoff.md §8 adopted and papers_fe.parquet carries.
Both are registered on the **Modal** backend in embedding_utils.MODAL_MODEL_KEYS (only
`qwen/qwen3-embedding-8b` is on the OpenRouter path), so both run through the already-
deployed `tiri-embeddings` app. Cost, from that report's §9 extrapolation of this exact
pairing to ~100k papers: a few dollars combined, one-time.

CACHE LAYOUT — and why it is per-collection
-------------------------------------------
    benchset_v1_<use_case_key>_<safe_model>_papers.parquet   paper_id, embedding
    benchset_v1_<safe_model>_usecases.parquet                use_case_key, embedding

The existing cache is one `<safe_model>_papers.parquet` for the whole corpus, keyed by
`paper_id`. That does NOT work here: `paper_id` is a content fingerprint and is unique
only *within* a collection — 3,137 papers appear in more than one, and 130 of them carry
`positive` under one question and `negative` under another. One pooled file keyed on
`paper_id` would collapse those into a single ambiguous row and a `.loc[]` lookup would
fan out. Splitting per collection keeps `paper_id` a valid key inside each file, and
suits the notebooks too — they work one silo at a time, and never want all 181k vectors
in memory at once.

The `_usecases.parquet` side keeps the existing schema exactly, so the read pattern in
run_embedding_recall_comparison.py `build_variants` carries over unchanged.

Brief text is built from the same five columns the bake-off used
(`use_case_name problem_statement objective domain_industry domain_application`), joined
the same way — a different join here would make cosine-to-brief incomparable with every
number already measured on the TIRI corpus.

RESUME
------
Two levels, because the largest collection is 48,343 papers:
  - a finished collection is skipped outright (its parquet exists and is complete);
  - inside a collection, each chunk is written to `_partial/` as it lands, so an
    interrupted run resumes at ~CHUNK_SIZE granularity instead of restarting the
    collection.
Re-running after a completed run is a no-op.

Usage:
    python scripts/embed_benchsets.py                      # everything, both models
    python scripts/embed_benchsets.py --collections roadfreight_metareview   # smoke test
    python scripts/embed_benchsets.py --models jasper      # one model
    python scripts/embed_benchsets.py --dry-run            # what's missing, embed nothing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_utils import MODEL_CONFIGS, embed_papers, embed_texts  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BENCHSETS = REPO / "data" / "benchsets_v1"
CACHE = REPO / "data" / "processed" / "embeddings_cache"
PARTIAL = CACHE / "_partial"

MODELS = {
    "jasper": "infgrad/Jasper-Token-Compression-600M",
    "qwen4b": "Qwen/Qwen3-Embedding-4B",
}

# The bake-off's own use-case query text, field-for-field — see this module's docstring.
USE_CASE_COLS = ["use_case_name", "problem_statement", "objective",
                 "domain_industry", "domain_application"]

# Big enough that per-call overhead is amortised, small enough that one lost chunk is a
# minute of work and the return payload stays well clear of Modal's limits (500 x 2560
# float64 ~ 10 MB).
CHUNK_SIZE = 500


def safe_name(model_name: str) -> str:
    """The cache-filename form of a model name — identical to the one
    notebooks/experiments/wf_embedding_model_bakeoff.ipynb already writes with."""
    return model_name.replace("/", "__").replace(":", "_")


def papers_path(use_case_key: str, model_name: str) -> Path:
    return CACHE / f"benchset_v1_{use_case_key}_{safe_name(model_name)}_papers.parquet"


def usecases_path(model_name: str) -> Path:
    return CACHE / f"benchset_v1_{safe_name(model_name)}_usecases.parquet"


def collection_files() -> dict[str, Path]:
    """use_case_key -> parquet path. Keyed off the filename, then verified against the
    file's own use_case_key column when it's read — a filename is a guess until checked."""
    return {p.stem: p for p in sorted(BENCHSETS.glob("*.parquet")) if p.stem != "briefs"}


def _write_vectors(path: Path, ids: list[str], vectors: np.ndarray, id_col: str) -> None:
    """One row per id, `embedding` as float32.

    float32, not the float64 the existing cache holds: `.tolist()` on a numpy array
    yields Python floats, so the 2,873-paper cache is 8 bytes per dimension. At 181,199
    papers x 4,608 combined dimensions that convention would cost 6.7 GB for no precision
    that survives a cosine. Keeping numpy arrays lets pyarrow store list<float>.
    """
    frame = pd.DataFrame({id_col: ids, "embedding": list(vectors.astype(np.float32))})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def embed_collection(model_key: str, model_name: str, use_case_key: str,
                     df: pd.DataFrame, dry_run: bool) -> str:
    """Embed one collection's papers, resuming from any chunks already on disk.

    Returns a one-line status for the caller to print.
    """
    out = papers_path(use_case_key, model_name)
    if out.exists():
        n_cached = len(pd.read_parquet(out, columns=["paper_id"]))
        if n_cached == len(df):
            return f"cached ({n_cached} papers)"
        # A short file is a half-written one from a killed process, not a valid cache.
        print(f"    {out.name}: {n_cached} rows but collection has {len(df)} — rebuilding")
        out.unlink()

    stem = f"benchset_v1_{use_case_key}_{safe_name(model_name)}"
    done: dict[str, np.ndarray] = {}
    for part in sorted(PARTIAL.glob(f"{stem}.part*.parquet")):
        chunk = pd.read_parquet(part)
        done.update(zip(chunk["paper_id"], chunk["embedding"]))

    todo = df[~df["paper_id"].isin(done)]
    if dry_run:
        return f"WOULD EMBED {len(todo)} of {len(df)} papers ({len(done)} already in _partial)"
    if len(done):
        print(f"    resuming: {len(done)} already done, {len(todo)} to go")

    PARTIAL.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for offset in range(0, len(todo), CHUNK_SIZE):
        chunk = todo.iloc[offset:offset + CHUNK_SIZE]
        vectors, _ = embed_papers(
            model_name,
            chunk["title"].fillna("").tolist(),
            chunk["abstract"].fillna("").tolist(),
        )
        # Sequence number keyed to position in the collection, so a resumed run's parts
        # never collide with an earlier run's.
        part = PARTIAL / f"{stem}.part{len(done):07d}.parquet"
        _write_vectors(part, chunk["paper_id"].tolist(), vectors, "paper_id")
        done.update(zip(chunk["paper_id"], vectors))

        elapsed = time.monotonic() - started
        rate = (len(done) - (len(df) - len(todo))) / elapsed if elapsed else 0
        print(f"      {len(done)}/{len(df)} papers  ({rate:.0f}/s)", flush=True)

    # Reindex to the collection's own row order before writing — the notebooks join on
    # paper_id, but a stable order makes the file diffable and the read cheaper.
    ordered = np.vstack([np.asarray(done[pid]) for pid in df["paper_id"]])
    _write_vectors(out, df["paper_id"].tolist(), ordered, "paper_id")
    for part in PARTIAL.glob(f"{stem}.part*.parquet"):
        part.unlink()
    return f"embedded {len(df)} papers in {time.monotonic() - started:.0f}s"


def embed_briefs(model_key: str, model_name: str, briefs: pd.DataFrame, dry_run: bool) -> str:
    """One vector per use case, for cosine-to-brief. Same schema and same query text as
    the existing <safe_model>_usecases.parquet files."""
    out = usecases_path(model_name)
    cached: dict[str, list] = {}
    if out.exists():
        prev = pd.read_parquet(out)
        cached = dict(zip(prev["use_case_key"], prev["embedding"]))

    texts = {
        row["use_case_key"]: " ".join(
            str(row[c]) for c in USE_CASE_COLS if pd.notna(row[c]) and str(row[c]).strip()
        )
        for _, row in briefs.iterrows()
    }
    missing = [uc for uc in texts if uc not in cached]
    if not missing:
        return f"cached ({len(cached)} briefs)"
    if dry_run:
        return f"WOULD EMBED {len(missing)} briefs"

    cfg = MODEL_CONFIGS[model_name]
    vectors = embed_texts(
        model_name, [texts[uc] for uc in missing],
        prefix=cfg["query_prefix"], is_query=True,
    )
    cached.update(zip(missing, [v.astype(np.float32) for v in vectors]))
    keys = list(cached)
    _write_vectors(out, keys, np.vstack([np.asarray(cached[k]) for k in keys]), "use_case_key")
    return f"embedded {len(missing)} briefs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", default=",".join(MODELS),
                        help=f"comma-separated subset of {list(MODELS)}")
    parser.add_argument("--collections", default="",
                        help="comma-separated use_case_keys (default: all 28)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is missing without embedding anything")
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(model_keys) - set(MODELS)
    if unknown:
        parser.error(f"unknown model key(s) {unknown}; known: {list(MODELS)}")

    files = collection_files()
    if args.collections:
        wanted = [c.strip() for c in args.collections.split(",") if c.strip()]
        missing = set(wanted) - set(files)
        if missing:
            parser.error(f"no such collection(s): {sorted(missing)}")
        files = {k: files[k] for k in wanted}

    briefs = pd.read_parquet(BENCHSETS / "briefs.parquet")
    CACHE.mkdir(parents=True, exist_ok=True)

    for model_key in model_keys:
        model_name = MODELS[model_key]
        print(f"\n{'=' * 78}\n{model_key}  ({model_name})\n{'=' * 78}")
        print(f"  briefs: {embed_briefs(model_key, model_name, briefs, args.dry_run)}")

        for use_case_key, path in files.items():
            df = pd.read_parquet(path, columns=["paper_id", "use_case_key", "title", "abstract"])
            actual = set(df["use_case_key"].unique())
            if actual != {use_case_key}:
                raise ValueError(
                    f"{path.name}: filename says use_case_key={use_case_key!r} but the file "
                    f"contains {sorted(actual)!r} — the cache key would be wrong, so stopping "
                    "rather than writing vectors under a name that doesn't match the data."
                )
            if not df["paper_id"].is_unique:
                raise ValueError(
                    f"{path.name}: paper_id is not unique within the collection, which the "
                    "per-collection cache layout depends on (see this module's docstring)."
                )
            print(f"  {use_case_key:34s} {len(df):6d} papers  "
                  f"{embed_collection(model_key, model_name, use_case_key, df, args.dry_run)}",
                  flush=True)


if __name__ == "__main__":
    main()
