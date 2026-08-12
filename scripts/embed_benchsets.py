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

# Brief variants, for the provenance ablation in 03_eda_full_benchset_v1.ipynb §8.4.
#
# The worry these answer: cosine-to-brief scores far higher on this corpus than on TIRI,
# and one obvious explanation is that `objective` holds the review's ABSTRACT — written
# after screening finished, describing the studies that got included. Every other field is
# a title, a name, a scope descriptor or a term list, all of which a customer genuinely has
# before screening starts. Dropping `objective` and re-scoring measures how much of the
# advantage depends on text that could not exist at t=0.
#
# `pre_screening` is the one that matters operationally: it is the closest thing here to a
# brief someone could actually write on day one.
BRIEF_VARIANTS = {
    "full": USE_CASE_COLS,
    "no_objective": ["use_case_name", "problem_statement",
                     "domain_industry", "domain_application"],
    "pre_screening": ["use_case_name", "problem_statement",
                      "domain_industry", "domain_application",
                      "terms_must_include", "terms_nice_to_have", "terms_exclude"],
}

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


def load_paper_vectors(df: pd.DataFrame, model_key: str, *,
                       id_col: str = "paper_id",
                       use_case_col: str = "use_case_key") -> np.ndarray:
    """Cached paper vectors for `df`'s rows, as one (len(df), dim) float32 matrix in df's
    own row order.

    This is the read side of the per-collection cache layout described at the top of this
    module, and it exists so that no downstream notebook has to know that layout. It is
    also the reason 04_feature_engineering_benchset_v1.ipynb does NOT explode embeddings
    into columns the way 04_feature_engineering.ipynb does: at 175k rows x 4,608 combined
    dimensions that would be ~3.2 GB written three times over, duplicating vectors this
    cache already holds. The feature tables stay small and join here at fit time instead.

    `paper_id` is unique only WITHIN a collection, so the lookup is keyed on the
    (use_case_key, paper_id) pair — a global paper_id -> vector dict would silently
    collide across the 2,102 papers that appear in more than one collection.

    Raises rather than filling if any row has no cached vector: a missing embedding means
    the cache is stale for this input, and a zero row would quietly poison every cosine
    computed from it.
    """
    model_name = MODELS[model_key]
    matrix, filled = None, np.zeros(len(df), dtype=bool)
    positions = np.arange(len(df))

    for use_case_key, rows in df.groupby(use_case_col, sort=False):
        path = papers_path(use_case_key, model_name)
        if not path.exists():
            raise FileNotFoundError(
                f"{model_key}: no cached vectors for collection {use_case_key!r} at "
                f"{path}. Run `python scripts/embed_benchsets.py --models {model_key} "
                f"--collections {use_case_key}` first."
            )
        cached = pd.read_parquet(path).set_index("paper_id")["embedding"]
        wanted = rows[id_col]
        missing = wanted[~wanted.isin(cached.index)]
        if len(missing):
            raise KeyError(
                f"{model_key}/{use_case_key}: {len(missing)} paper_id(s) have no cached "
                f"vector, e.g. {sorted(missing)[:5]} — the cache is stale for this input."
            )
        block = np.vstack(cached.loc[wanted].to_numpy()).astype(np.float32)
        if matrix is None:
            matrix = np.empty((len(df), block.shape[1]), dtype=np.float32)
        elif block.shape[1] != matrix.shape[1]:
            raise ValueError(
                f"{model_key}: {use_case_key} cached at {block.shape[1]} dims but an "
                f"earlier collection cached at {matrix.shape[1]} — mixed model versions."
            )
        where = positions[df[use_case_col].to_numpy() == use_case_key]
        matrix[where] = block
        filled[where] = True

    if matrix is None:
        raise ValueError("empty frame — nothing to load vectors for")
    assert filled.all(), f"{(~filled).sum()} rows never filled — groupby missed them"
    return matrix


def load_brief_vectors(model_key: str, variant: str = "full") -> dict[str, np.ndarray]:
    """use_case_key -> brief vector, for one of BRIEF_VARIANTS.

    `full` is what the bake-off scored and what cosine-to-brief has always meant here.
    `pre_screening` is the variant a customer could actually write at t=0; §8.4 of
    03_eda_full_benchset_v1.ipynb measured the gap between them at ~0.005 ROC-AUC against
    a ~0.03 noise floor, which is why both are worth carrying downstream.
    """
    if variant not in BRIEF_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; have {sorted(BRIEF_VARIANTS)}")
    frame = pd.read_parquet(usecases_path(MODELS[model_key]))
    frame = frame[frame["variant"] == variant]
    if frame.empty:
        raise ValueError(
            f"{model_key}: no {variant!r} briefs cached in "
            f"{usecases_path(MODELS[model_key])} — re-run the runner to add them."
        )
    return {k: np.asarray(v, dtype=np.float32)
            for k, v in zip(frame["use_case_key"], frame["embedding"])}


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
                     df: pd.DataFrame, dry_run: bool,
                     shard: tuple[int, int] | None = None) -> str:
    """Embed one collection's papers, resuming from any chunks already on disk.

    `shard=(i, n)` takes only every n-th row, so the two collections big enough to
    dominate the whole job (48,343 and 38,114 papers) can be split across concurrent
    processes instead of being a serial tail. Shards are disjoint by construction and each
    writes its own part files; whichever finishes last finds every id on disk and
    consolidates. The others exit saying so.

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

    mine = df if shard is None else df.iloc[shard[0]::shard[1]]
    todo = mine[~mine["paper_id"].isin(done)]
    if dry_run:
        return f"WOULD EMBED {len(todo)} of {len(mine)} papers ({len(done)} already in _partial)"
    if len(done):
        print(f"    resuming: {len(done)} of {len(df)} already done, {len(todo)} in this shard")

    # Part files must not collide between concurrent shards: two processes computing
    # `len(done)` at the same moment would otherwise pick the same name and one would
    # overwrite the other's vectors.
    token = "all" if shard is None else f"{shard[0]}of{shard[1]}"

    PARTIAL.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for seq, offset in enumerate(range(0, len(todo), CHUNK_SIZE)):
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
        part = PARTIAL / f"{stem}.part{seq:05d}_{token}.parquet"
        _write_vectors(part, chunk["paper_id"].tolist(), vectors, "paper_id")
        done.update(zip(chunk["paper_id"], vectors))

        elapsed = time.monotonic() - started
        n_this_run = (seq + 1) * CHUNK_SIZE
        print(f"      {len(done)}/{len(df)} papers  "
              f"({n_this_run / elapsed if elapsed else 0:.0f}/s this process)", flush=True)

    # Re-read _partial before deciding. `done` only reflects what existed when this
    # process STARTED plus what it embedded itself, so concurrent shards would each finish
    # holding their own slice and none would ever see a complete collection — nothing
    # would consolidate and the run would end with the work done but no output file.
    for part in sorted(PARTIAL.glob(f"{stem}.part*.parquet")):
        chunk = pd.read_parquet(part)
        done.update(zip(chunk["paper_id"], chunk["embedding"]))

    # Only the process that can see every id may consolidate. With shards running
    # concurrently the others simply stop here and leave _partial alone.
    if len(done) < len(df):
        return (f"shard done ({len(todo)} embedded); {len(done)}/{len(df)} of the "
                "collection is on disk — another shard will consolidate")

    # Reindex to the collection's own row order before writing — the notebooks join on
    # paper_id, but a stable order makes the file diffable and the read cheaper.
    ordered = np.vstack([np.asarray(done[pid]) for pid in df["paper_id"]])
    _write_vectors(out, df["paper_id"].tolist(), ordered, "paper_id")
    for part in PARTIAL.glob(f"{stem}.part*.parquet"):
        part.unlink(missing_ok=True)  # a sibling shard may have consolidated first
    return f"embedded {len(df)} papers in {time.monotonic() - started:.0f}s"


def embed_briefs(model_key: str, model_name: str, briefs: pd.DataFrame, dry_run: bool) -> str:
    """One vector per (use case, brief variant), for cosine-to-brief and the §8.4 ablation.

    The `variant` column is an addition to the TIRI cache's <safe_model>_usecases.parquet
    schema; readers wanting the production brief filter `variant == "full"`.
    """
    out = usecases_path(model_name)
    if out.exists() and "variant" in pd.read_parquet(out).columns:
        return f"cached ({len(BRIEF_VARIANTS)} variants x {len(briefs)} briefs)"
    if dry_run:
        return f"WOULD EMBED {len(BRIEF_VARIANTS) * len(briefs)} brief variants"

    def field_text(row, column):
        value = row[column]
        if isinstance(value, (list, np.ndarray)):
            return " ; ".join(str(v) for v in value if str(v).strip())
        return str(value) if pd.notna(value) and str(value).strip() else ""

    cfg = MODEL_CONFIGS[model_name]
    keys, variants, vectors = [], [], []
    for variant, columns in BRIEF_VARIANTS.items():
        texts = [
            " ".join(t for t in (field_text(row, c) for c in columns) if t)
            for _, row in briefs.iterrows()
        ]
        embedded = embed_texts(model_name, texts, prefix=cfg["query_prefix"], is_query=True)
        keys.extend(briefs["use_case_key"].tolist())
        variants.extend([variant] * len(briefs))
        vectors.extend(v.astype(np.float32) for v in embedded)

    frame = pd.DataFrame({"use_case_key": keys, "variant": variants,
                          "embedding": list(np.vstack(vectors).astype(np.float32))})
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    return f"embedded {len(BRIEF_VARIANTS)} variants x {len(briefs)} briefs"


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
            expected_rows = len(files) * len(BRIEF_VARIANTS)
            if len(briefs) != expected_rows:
                problems.append(f"{briefs_path.name}: {len(briefs)} rows, expected "
                                f"{expected_rows} ({len(BRIEF_VARIANTS)} variants x {len(files)})")
            if vectors.shape[1] != expected_dim:
                problems.append(f"{briefs_path.name}: dim {vectors.shape[1]} != {expected_dim}")
            if set(briefs["variant"]) != set(BRIEF_VARIANTS):
                problems.append(f"{briefs_path.name}: variants {sorted(set(briefs['variant']))} "
                                f"!= {sorted(BRIEF_VARIANTS)}")
            if briefs.duplicated(subset=["use_case_key", "variant"]).any():
                problems.append(f"{briefs_path.name}: duplicate (use_case_key, variant)")

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
    parser.add_argument("--shard", default="",
                        help="'i/n' — take only every n-th row, to split one big collection "
                             "across concurrent processes (run all n, any order)")
    args = parser.parse_args()

    shard = None
    if args.shard:
        try:
            i, n = (int(part) for part in args.shard.split("/"))
        except ValueError:
            parser.error(f"--shard must look like 'i/n', got {args.shard!r}")
        if not 0 <= i < n:
            parser.error(f"--shard index must satisfy 0 <= i < n, got {args.shard!r}")
        shard = (i, n)

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
            status = embed_collection(model_key, model_name, use_case_key, df,
                                      args.dry_run, shard)
            print(f"  {use_case_key:34s} {len(df):6d} papers  {status}", flush=True)


if __name__ == "__main__":
    main()
