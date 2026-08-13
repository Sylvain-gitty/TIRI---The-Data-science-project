"""Build the set-A ensemble feature table with both brief variants side by side.

Why a purpose-built table
-------------------------
`benchset_v1_large_set_a.parquet` already carries `lex_*` and `cos_brief_*` built from the
**supplied** brief. Testing whether the induced rule set (`wf_llm_setA_rules_v1.json`) helps
the ensemble needs the same columns rebuilt from the **induced** brief, and needs both to be
built identically — the shipped `lex_*` were computed with pool IDF over the full
62,229-row collections, so mixing them with a freshly-built induced arm would confound the
brief swap with a pool change. Both arms here are rebuilt over the same sampled rows.

Rows: the 9,993-row case-control sample, not all 62,229. Two reasons, one practical and one
principled. Practical: `modal_ensemble_experiments.py` records ~300 rows x 4,600 columns at
100 CatBoost iterations twice exceeding a 1,800s timeout, and `brouwer_2019` alone is 37,401
rows — the full corpus is not fittable at this width. Principled: every other set-A number
is on this sample, `validate_benchset_sampling.py` gates it, and all arms share the identical
rows so the brief comparison is **paired**.

What that costs, stated plainly: training on a case-control sample raises the base rate the
model sees (4-28% by silo, against a population 0.2-12%). Ranking metrics are insensitive to
the prior and the comparison is paired, so the brief effect is measured cleanly. Absolute
calibrated probabilities are not population values and are not reported as such.

Three feature variants, so the full swap is separable from the part that moved:

    supplied      supplied lex  + supplied cos  + metadata + embeddings
    induced_lex   induced  lex  + supplied cos  + metadata + embeddings
    induced_all   induced  lex  + induced  cos  + metadata + embeddings

`wf_llm_benchset_a_induced_features.md` measured the cosine as unmoved by the brief swap
(+0.004 / -0.005), so `induced_lex` is the arm carrying the hypothesis and `induced_all` is
the honest end-to-end version.

Usage:
    python scripts/build_setA_ensemble_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchset_loader import SET_A, case_control_sample, load_set_a  # noqa: E402
from compare_setA_induced_brief import (  # noqa: E402
    BRIEF_VECTORS, EMB_MODELS, cosine, embed_induced_briefs, induced_frame,
)
from lexical_features import BRIEF_KEYS, build_lexical_features  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RULES = REPO / "reports" / "wf_llm_setA_rules_v1.json"
OUT = REPO / "data" / "processed" / "setA_ensemble_features.parquet"

# The 10-column BM25 subset run_ensemble_candidate.py ships, not the full 22-column block:
# "BM25 alone is 0.637 of the full block's 0.642 LOGO ROC-AUC ... ship ~10 columns, not 22,
# given how small each silo is".
LEX_KEEP = [f"bm25_{k}" for k in BRIEF_KEYS] + [f"rank_bm25_{k}" for k in BRIEF_KEYS]
# Set A carries year / n_authors / citation_count. `paper_age` and `has_abstract` are in the
# TIRI METADATA_COLS but are not in this export, and are NOT synthesised — a missing column
# is missing, not zero.
META_COLS = ["year", "n_authors", "citation_count"]
KEY_COLS = ["row_key", "paper_id", "use_case_key", "first_author", "y", "w", "split"]


def pct_rank(values: np.ndarray) -> np.ndarray:
    """Within-pool percentile rank, matching lexical_features._pct_rank's convention."""
    order = values.argsort()
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(len(values) - 1, 1)


def main() -> None:
    fields = json.loads(RULES.read_text())["fields"]
    base = load_set_a()
    sample = case_control_sample(base).reset_index(drop=True)
    print(f"{len(sample):,} rows / {sample.use_case_key.nunique()} silos / "
          f"{int(sample.y.sum()):,} positive")

    # `load_set_a` deliberately reads only keys, baselines and text out of a 4,648-column
    # file; the metadata block is not part of the cold-start baseline so it is fetched here.
    meta = pd.read_parquet(SET_A, columns=["row_key", "use_case_key", *META_COLS])
    out = sample[KEY_COLS].merge(meta, on=["row_key", "use_case_key"], how="left",
                                 validate="one_to_one")
    if out[META_COLS].isna().all().any():
        raise RuntimeError(f"metadata join produced an all-null column: {META_COLS}")

    # ---- lexical block, both briefs, identical construction ----
    for tag, frame in (("lex", sample), ("lexind", induced_frame(sample, fields))):
        block = build_lexical_features(frame)
        for col in LEX_KEEP:
            out[f"{tag}_{col}"] = block[col].to_numpy()
        print(f"  {tag}: {len(LEX_KEEP)} columns")

    # ---- cosine-to-brief, both briefs ----
    bv = embed_induced_briefs(fields).set_index(["model", "use_case_key"]).embedding
    names = pq.ParquetFile(SET_A).schema_arrow.names
    emb_frames = []
    for short in EMB_MODELS:
        cols = [n for n in names if n.startswith(f"emb_{short}_")]
        if not cols:
            continue
        embs = pd.read_parquet(SET_A, columns=["row_key", "use_case_key", *cols])
        joined = sample[["row_key", "use_case_key"]].merge(
            embs, on=["row_key", "use_case_key"], how="left", validate="one_to_one")
        # supplied side is read straight off disk - it is the shipped column, and unlike the
        # lexical block a cosine has no pool-dependent term, so there is nothing to rebuild.
        out[f"cos_brief_{short}"] = sample[f"cos_brief_{short}"].to_numpy()
        out[f"rank_cos_brief_{short}"] = sample[f"rank_cos_brief_{short}"].to_numpy()
        ind = np.full(len(sample), np.nan)
        for uc, g in joined.groupby("use_case_key"):
            idx = g.index.to_numpy()
            P = g[cols].to_numpy(np.float64)
            c = cosine(P, np.asarray(bv.loc[(short, uc)], dtype=np.float64))
            ind[idx] = c
        out[f"cosind_brief_{short}"] = ind
        r = np.full(len(sample), np.nan)
        for uc, g in sample.groupby("use_case_key"):
            idx = g.index.to_numpy()
            r[idx] = pct_rank(ind[idx])
        out[f"cosind_rank_brief_{short}"] = r
        for col in cols:
            out[col] = joined[col].to_numpy(np.float32)
        emb_frames.append(len(cols))
        del embs, joined
        print(f"  {short}: cosine + {len(cols)} embedding columns")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    size_mb = OUT.stat().st_size / 1e6
    print(f"\nwrote {OUT}  ({out.shape[0]:,} x {out.shape[1]:,}, {size_mb:.0f} MB)")

    # Sanity: the induced cosine must NOT be identical to the supplied one (that would mean
    # the brief swap silently did nothing), and must not be wildly decorrelated either.
    for short in EMB_MODELS:
        a, b = out[f"cos_brief_{short}"], out[f"cosind_brief_{short}"]
        if b.notna().any():
            print(f"  {short}: supplied-vs-induced cosine corr {np.corrcoef(a, b)[0,1]:.3f}, "
                  f"max|diff| {np.abs(a - b).max():.3f}")


if __name__ == "__main__":
    main()
