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
    python scripts/embed_benchsets.py --verify             # check what's written, exit 1 on any problem

The job parallelises by running several invocations over disjoint --collections at once;
Modal scales out a container per concurrent call, and the skip/resume logic keeps them
from treading on each other. Keep the sets disjoint — two processes on one collection
duplicate the work and race on consolidation.
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
# Asserted by --verify rather than assumed. These are the widths papers_fe.parquet
# already carries (04_feature_engineering.ipynb's own expected_widths check), so a
# mismatch means the model changed under us, not that this constant is stale.
EXPECTED_DIMS = {"jasper": 2048, "qwen4b": 2560}

# The bake-off's own use-case query text, field-for-field — see this module's docstring.
USE_CASE_COLS = ["use_case_name", "problem_statement", "objective",
                 "domain_industry", "domain_application"]

# Big enough that per-call overhead is amortised, small enough that one lost chunk is a
# minute of work and the return payload stays well clear of Modal's limits (500 x 2560
# float64 ~ 10 MB).
CHUNK_SIZE = 500

# Abstract character budget. MEASURED, not guessed: title+abstract across this corpus has
# a median of 1,450 characters and a 99.9th percentile of 5,871 — but a 301-row tail runs
# to 30,177, which is a full-text dump that escaped into an abstract field, not an
# abstract. Those rows are what make the GPU run out of memory: Jasper batches 32
# documents at a time server-side, and 32 x 30k characters of attention does not fit on an
# L4 (confirmed by a torch.OutOfMemoryError on synergy_walker_2018 and three others).
#
# 4,000 characters keeps 99.8% of the corpus completely untouched and costs the rest a
# tail that a screening decision does not depend on. This is a truncation that would
# happen anyway — sentence-transformers silently truncates at the model's max_seq_length —
# so the choice here is only whether it is visible and reproducible. The TITLE is never
# truncated, only the abstract.
MAX_ABSTRACT_CHARS = 4000

# When a chunk fails anyway, retry it in pieces this size. Below the models' server-side
# batch_size (32 for Jasper, 16 for Qwen3-4B), a chunk IS the batch, so a smaller chunk is
# the one lever a client has over peak GPU memory.
OOM_RETRY_CHUNK = 8


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
        titles = chunk["title"].fillna("").tolist()
        abstracts = chunk["abstract"].fillna("").str.slice(0, MAX_ABSTRACT_CHARS).tolist()
        try:
            vectors, _ = embed_papers(model_name, titles, abstracts)
        except Exception as exc:  # noqa: BLE001 — see MAX_ABSTRACT_CHARS
            # Almost always a GPU OOM on an unusually long batch. Retrying the same chunk
            # unchanged would fail identically, so retry it in pieces small enough to be
            # the server-side batch themselves. Re-raised if that fails too: an embedding
            # that cannot be produced is a finding, not a row to quietly skip.
            print(f"      chunk at {offset} failed ({type(exc).__name__}), "
                  f"retrying in {OOM_RETRY_CHUNK}-row pieces", flush=True)
            pieces = [
                embed_papers(model_name, titles[i:i + OOM_RETRY_CHUNK],
                             abstracts[i:i + OOM_RETRY_CHUNK])[0]
                for i in range(0, len(titles), OOM_RETRY_CHUNK)
            ]
            vectors = np.vstack(pieces)
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


def verify(files: dict[str, Path], model_keys: list[str]) -> int:
    """Check every written cache file against the source collection it claims to cover.

    Worth its own pass rather than trusting the run's exit code: a half-written parquet
    from a killed process, a silently truncated collection, or an all-zero vector from a
    model that loaded but did not run are all things a green run can leave behind, and all
    of them would surface much later as a confusing notebook error.
    """
    problems: list[str] = []
    n_checked = n_missing = 0

    for model_key in model_keys:
        model_name = MODELS[model_key]
        expected_dim = EXPECTED_DIMS[model_key]

        briefs_path = usecases_path(model_name)
        if not briefs_path.exists():
            n_missing += 1
        else:
            briefs = pd.read_parquet(briefs_path)
            vectors = np.vstack(briefs["embedding"].to_numpy())
            if len(briefs) != len(files):
                problems.append(f"{briefs_path.name}: {len(briefs)} briefs, expected {len(files)}")
            if vectors.shape[1] != expected_dim:
                problems.append(f"{briefs_path.name}: dim {vectors.shape[1]} != {expected_dim}")
            if not briefs["use_case_key"].is_unique:
                problems.append(f"{briefs_path.name}: duplicate use_case_key")

        for use_case_key, source_path in files.items():
            path = papers_path(use_case_key, model_name)
            if not path.exists():
                n_missing += 1
                continue
            n_checked += 1
            cached = pd.read_parquet(path)
            vectors = np.vstack(cached["embedding"].to_numpy())
            source_ids = pd.read_parquet(source_path, columns=["paper_id"])["paper_id"]

            if vectors.shape[1] != expected_dim:
                problems.append(f"{path.name}: dim {vectors.shape[1]} != {expected_dim}")
            if vectors.dtype != np.float32:
                problems.append(f"{path.name}: dtype {vectors.dtype} != float32")
            if not cached["paper_id"].is_unique:
                problems.append(f"{path.name}: paper_id is not unique")
            if not np.isfinite(vectors).all():
                problems.append(f"{path.name}: contains NaN or inf")
            n_zero = int((np.linalg.norm(vectors, axis=1) == 0).sum())
            if n_zero:
                problems.append(f"{path.name}: {n_zero} all-zero vectors")
            if set(source_ids) != set(cached["paper_id"]):
                problems.append(
                    f"{path.name}: paper_id set differs from {source_path.name} "
                    f"({len(cached)} cached vs {len(source_ids)} source rows)"
                )

    print(f"Checked {n_checked} collection files; {n_missing} not yet written.")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("All checked files pass: dims, float32, unique and complete paper_id sets, "
          "finite, no zero vectors.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", default=",".join(MODELS),
                        help=f"comma-separated subset of {list(MODELS)}")
    parser.add_argument("--collections", default="",
                        help="comma-separated use_case_keys (default: all 28)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is missing without embedding anything")
    parser.add_argument("--verify", action="store_true",
                        help="check already-written cache files and exit non-zero on any problem")
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

    if args.verify:
        raise SystemExit(verify(files, model_keys))

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
