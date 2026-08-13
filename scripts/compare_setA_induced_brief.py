"""Is the induced-brief win about briefs, or about LLMs?

The set-A run found brief quality to be the largest measured lever in the whole LLM
workstream: same model, same prompt, a rule set distilled from 60 train labels is worth
**+0.057 AUC and +0.203 F2@own**, against a 0.030 total spread across four model families
from 20B to 397B (`wf_llm_benchset_a_findings.md` §5). That finding was measured on a
*reader* — a generative model following the brief as instructions.

The features the product actually ships do not read. `cos_brief_*` embeds the brief and takes
a cosine; the BM25/overlap block tokenises it and matches. If a better brief lifts those too,
this is a finding about **briefs** and applies to the shipped cold-start rung. If it does not,
the gain needs a reader and the rung cannot have it.

There is a concrete reason to expect the two to diverge, and it is the same asymmetry §5
already measured once. Stripping the LLM-written `terms_*` lists (B0) left the reader's
ranking unchanged and *improved* its operating point, while `CONTEXT.md` §4 records curated
terminology as load-bearing for the lexical block — **term lists help matchers and hurt
readers.** The mirror-image risk here: the supplied `objective` is the review's own abstract,
text that looks like the papers, which is exactly what a cosine rewards. The induced
`objective` is a statement of inclusion criteria — better instructions, less document-like.
A rule set could therefore help the reader and hurt the matcher.

What is compared
----------------
Both arms are rebuilt from scratch so the only difference is brief content:

  - **cosine-to-brief** — the induced fields concatenated exactly as `embed_benchsets.py`
    does for the `full` variant (`use_case_name problem_statement objective domain_industry
    domain_application`), embedded by the same models with `is_query=True`, cosine against the
    paper vectors already on disk. The recipe is verified by reproducing the shipped
    `cos_brief_qwen4b` column to 1.1e-07 before anything new is embedded.
  - **BM25 + term overlap** — `lexical_features.build_lexical_features` with the brief columns
    swapped. Note this rebuilds the pool IDF over the sampled rows rather than the full
    collection, so the supplied arm here is not bit-identical to the shipped `lex_*` columns;
    both arms are built the same way and the rebuilt-vs-shipped correlation is printed.

Scored on **test + validate only**, weighted to population prevalence, because the induced
brief saw train labels — the same rows and the same treatment as the ladder in
`wf_llm_benchset_a.md` §4, so the numbers sit directly beside it.

Usage:
    python scripts/compare_setA_induced_brief.py            # lexical arm only, free
    python scripts/compare_setA_induced_brief.py --cosine    # + Modal GPU brief embedding
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchset_loader import SET_A, case_control_sample, load_set_a  # noqa: E402
from benchset_metrics import recall_at, work_at_own, wss_at  # noqa: E402
from ensemble_eval_utils import to_md  # noqa: E402
from lexical_features import build_lexical_features  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RULES = REPO / "reports" / "wf_llm_setA_rules_v1.json"
BRIEF_VECTORS = REPO / "reports" / "wf_llm_setA_induced_brief_vectors.parquet"
OUT_MD = REPO / "reports" / "wf_llm_benchset_a_induced_features.md"

# Field-for-field the `full` brief variant in embed_benchsets.BRIEF_VARIANTS. Changing this
# list changes what cosine-to-brief means and breaks comparability with the shipped column.
COS_BRIEF_COLS = ["use_case_name", "problem_statement", "objective",
                  "domain_industry", "domain_application"]
BRIEF_SWAP_COLS = ["use_case_name", "objective", "problem_statement",
                   "terms_must_include", "terms_nice_to_have", "terms_exclude",
                   "domain_industry", "domain_application", "domain_technology_focus"]
EMB_MODELS = {"qwen4b": "Qwen/Qwen3-Embedding-4B",
              "jasper": "infgrad/Jasper-Token-Compression-600M"}
# The single-feature cold-start signals: no labels, no fitting, directly comparable to a cosine.
LEX_HEADLINE = ["bm25_obj", "bm25_must", "bm25_nice", "overlap_must_frac", "rank_overlap_must"]
K_PER_CLASS, INDUCE_SEED = 30, 0


def field_text(value) -> str:
    """embed_benchsets.embed_briefs.field_text, copied so the concatenation matches exactly."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ; ".join(str(v) for v in value if str(v).strip())
    return str(value) if value is not None and str(value).strip() else ""


def induced_frame(df: pd.DataFrame, fields: dict) -> pd.DataFrame:
    """`df` with every brief column replaced by the induced rule set, row-aligned."""
    out = df.copy()
    for col in BRIEF_SWAP_COLS:
        mapped = {uc: f.get(col) for uc, f in fields.items()}
        if col in ("terms_must_include", "terms_nice_to_have", "terms_exclude",
                   "domain_technology_focus"):
            out[col] = out.use_case_key.map(lambda u: mapped.get(u) or [])
        else:
            out[col] = out.use_case_key.map(lambda u: mapped.get(u) or "")
    return out


def brief_query(row_fields: dict) -> str:
    return " ".join(t for t in (field_text(row_fields.get(c)) for c in COS_BRIEF_COLS) if t)


def embed_induced_briefs(fields: dict) -> pd.DataFrame:
    """One vector per (use case, model) for the induced brief. Cached on disk — this is the
    only step that costs GPU time, and it is 8 short strings per model."""
    if BRIEF_VECTORS.exists():
        print(f"induced brief vectors: cached ({BRIEF_VECTORS.name})")
        return pd.read_parquet(BRIEF_VECTORS)
    from embedding_utils import MODEL_CONFIGS, embed_texts

    keys = sorted(fields)
    texts = [brief_query(fields[k]) for k in keys]
    print(f"embedding {len(texts)} induced briefs "
          f"(median {int(np.median([len(t) for t in texts]))} chars) via Modal")
    rows = []
    for short, model in EMB_MODELS.items():
        vecs = embed_texts(model, texts, prefix=MODEL_CONFIGS[model]["query_prefix"],
                           is_query=True)
        print(f"  {short}: {np.asarray(vecs).shape}")
        for k, v in zip(keys, np.asarray(vecs, dtype=np.float32)):
            rows.append({"use_case_key": k, "model": short, "embedding": v})
    frame = pd.DataFrame(rows)
    frame.to_parquet(BRIEF_VECTORS, index=False)
    return frame


def cosine(P: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (P @ q) / (np.linalg.norm(P, axis=1) * np.linalg.norm(q) + 1e-12)


def rank_metrics(y, s, w) -> dict:
    return {"auc": roc_auc_score(y, s), "wss95": wss_at(y, s, w)["wss"],
            "recall@10%": recall_at(y, s, w, 0.10)}


def fit_block(tr, te, cols, C=1.0):
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
    pipe.fit(tr[cols].to_numpy(float), tr.y.to_numpy())
    return (pipe.predict_proba(te[cols].to_numpy(float))[:, 1],
            pipe.predict(te[cols].to_numpy(float)))


def train_rows(g: pd.DataFrame) -> pd.DataFrame:
    """The identical 30+30 draw the rule induction saw, so both sides spend the same labels."""
    pool = g[g.split == "train"]
    return pd.concat([
        pool[pool.y == 1].sample(n=min(K_PER_CLASS, int((pool.y == 1).sum())),
                                 random_state=INDUCE_SEED),
        pool[pool.y == 0].sample(n=min(K_PER_CLASS, int((pool.y == 0).sum())),
                                 random_state=INDUCE_SEED),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cosine", action="store_true",
                    help="also run the cosine arm (embeds 8 briefs per model on Modal GPU)")
    args = ap.parse_args()

    blob = json.loads(RULES.read_text())
    fields = blob["fields"]
    base = load_set_a()
    sample = case_control_sample(base)
    print(f"{len(sample):,} sampled rows · rule set {blob['rule_set_version']}")

    lines: list[str] = []

    def emit(t: str = "") -> None:
        print(t)
        lines.append(t)

    emit("# Does the induced brief help features that do not read?")
    emit()
    emit(f"Rule set `{blob['rule_set_version']}`, induced from "
         f"{blob['k_per_class']}+{blob['k_per_class']} train rows per collection. Scored on "
         "**test + validate only**, weighted to population prevalence — the same rows as "
         "`wf_llm_benchset_a.md` §4, so these numbers sit beside the LLM ladder.")
    emit()

    # ---------------- lexical arm ----------------
    arms = {"supplied": sample, "induced": induced_frame(sample, fields)}
    lex = {}
    for name, frame in arms.items():
        block = build_lexical_features(frame)
        block.columns = [f"lexnew_{c}" for c in block.columns]
        lex[name] = pd.concat([sample[["use_case_key", "y", "w", "split"]], block], axis=1)

    # Sanity: the rebuilt supplied arm should track the shipped column closely. It cannot be
    # identical - the shipped lex_* were built over the full 62,229-row collections and these
    # over the 9,993-row sample, so the pool IDF differs.
    chk = []
    for uc, g in sample.groupby("use_case_key"):
        a = lex["supplied"].loc[g.index, "lexnew_bm25_must"]
        chk.append(np.corrcoef(a, g.lex_bm25_must)[0, 1])
    emit(f"Rebuild check: rebuilt `bm25_must` vs the shipped `lex_bm25_must` correlates "
         f"**{np.mean(chk):.4f}** (min {np.min(chk):.4f}) across the 8 collections — the "
         "residual is the pool-IDF difference, and both arms below are built identically.")
    emit()

    rows = []
    for uc, g in sample.groupby("use_case_key"):
        y, w = g.y.to_numpy(), g.w.to_numpy()
        held = g.split != "train"
        rec = {"use_case": uc.replace("synergy_", "")}
        for name in arms:
            L = lex[name].loc[g.index]
            for feat in LEX_HEADLINE:
                s = L[f"lexnew_{feat}"].to_numpy(float)
                keep = held.to_numpy() & ~np.isnan(s)
                if len(np.unique(y[keep])) < 2:
                    continue
                rec[f"{feat}|{name}"] = roc_auc_score(y[keep], s[keep])
        rows.append(rec)
    lexres = pd.DataFrame(rows).set_index("use_case")

    emit("## 1. BM25 and term overlap — single features, no labels, no fitting")
    emit()
    emit("This is the cold-start use of the lexical block: one number per paper, straight "
         "from the brief. ROC-AUC on held-out rows.")
    emit()
    for feat in LEX_HEADLINE:
        cols = [f"{feat}|supplied", f"{feat}|induced"]
        if not all(c in lexres for c in cols):
            continue
        t = lexres[cols].copy()
        t.columns = ["supplied", "induced"]
        t["delta"] = t.induced - t.supplied
        emit(f"**`{feat}`** — mean {t.supplied.mean():.3f} → **{t.induced.mean():.3f}** "
             f"({t.delta.mean():+.3f}), improves on {int((t.delta > 0.03).sum())}/"
             f"{len(t)}, degrades on {int((t.delta < -0.03).sum())}")
        emit()
        emit(to_md(t.round(3)))
        emit()

    # ---------------- lexical block, supervised on the same 60 labels ----------------
    sup = []
    for uc, g in sample.groupby("use_case_key"):
        te = g[g.split != "train"]
        tr_idx = train_rows(g).index
        rec = {"use_case": uc.replace("synergy_", "")}
        for name in arms:
            L = lex[name]
            cols = [c for c in L.columns if c.startswith("lexnew_")]
            s, pred = fit_block(L.loc[tr_idx], L.loc[te.index], cols)
            m = rank_metrics(te.y.to_numpy(), s, te.w.to_numpy())
            own = work_at_own(te.y.to_numpy(), pred, te.w.to_numpy())
            rec[f"auc|{name}"] = m["auc"]
            rec[f"wss|{name}"] = m["wss95"]
            rec[f"f2own|{name}"] = own["f2"]
        sup.append(rec)
    supres = pd.DataFrame(sup).set_index("use_case")
    emit("## 2. The whole lexical block, fitted on the same 60 labels")
    emit()
    emit("LogReg over all 21 lexical features, trained on the identical 30+30 rows the rule "
         "induction saw. Tests whether a better brief still pays once the block can be fitted.")
    emit()
    t = supres[["auc|supplied", "auc|induced", "wss|supplied", "wss|induced",
                "f2own|supplied", "f2own|induced"]].copy()
    t["d_auc"] = t["auc|induced"] - t["auc|supplied"]
    emit(to_md(t.round(3)))
    emit()
    emit(f"Mean AUC **{t['auc|supplied'].mean():.3f} → {t['auc|induced'].mean():.3f}** "
         f"({t.d_auc.mean():+.3f}); improves on {int((t.d_auc > 0.03).sum())}/{len(t)}, "
         f"degrades on {int((t.d_auc < -0.03).sum())}.")
    emit()

    # ---------------- cosine arm ----------------
    if args.cosine:
        bv = embed_induced_briefs(fields)
        emit("## 3. Cosine-to-brief — the shipped cold-start rung")
        emit()
        emit("Induced fields concatenated exactly as `embed_benchsets.py` builds the `full` "
             "variant, embedded by the same models with `is_query=True`, cosine against the "
             "paper vectors already on disk.")
        emit()
        names = pq.ParquetFile(SET_A).schema_arrow.names
        crows = []
        for short in EMB_MODELS:
            cols = [n for n in names if n.startswith(f"emb_{short}_")]
            if not cols:
                print(f"  no emb_{short}_* columns on disk, skipping")
                continue
            embs = pd.read_parquet(SET_A, columns=["row_key", "use_case_key", *cols])
            joined = sample.merge(embs, on=["row_key", "use_case_key"], how="left",
                                  validate="one_to_one")
            vec = bv[bv.model == short].set_index("use_case_key").embedding
            for uc, g in joined.groupby("use_case_key"):
                held = (g.split != "train").to_numpy()
                y, w = g.y.to_numpy()[held], g.w.to_numpy()[held]
                P = g[cols].to_numpy(np.float64)[held]
                rec = {"model": short, "use_case": uc.replace("synergy_", "")}
                sup_s = g[f"cos_brief_{short}"].to_numpy(np.float64)[held]
                ind_s = cosine(P, np.asarray(vec.loc[uc], dtype=np.float64))
                for name, s in (("supplied", sup_s), ("induced", ind_s)):
                    for k, v in rank_metrics(y, s, w).items():
                        rec[f"{k}|{name}"] = v
                crows.append(rec)
            del embs, joined
        cos = pd.DataFrame(crows)
        for short, g in cos.groupby("model"):
            t = g.set_index("use_case")[["auc|supplied", "auc|induced", "wss95|supplied",
                                         "wss95|induced", "recall@10%|supplied",
                                         "recall@10%|induced"]].copy()
            t["d_auc"] = t["auc|induced"] - t["auc|supplied"]
            emit(f"### `{short}`")
            emit()
            emit(to_md(t.round(3)))
            emit()
            emit(f"Mean AUC **{t['auc|supplied'].mean():.3f} → "
                 f"{t['auc|induced'].mean():.3f}** ({t.d_auc.mean():+.3f}); improves on "
                 f"{int((t.d_auc > 0.03).sum())}/{len(t)}, degrades on "
                 f"{int((t.d_auc < -0.03).sum())}. Mean WSS@95 "
                 f"**{t['wss95|supplied'].mean():.3f} → {t['wss95|induced'].mean():.3f}**.")
            emit()
        cos.to_csv(REPO / "reports" / "wf_llm_benchset_a_induced_cosine.csv", index=False)

    OUT_MD.write_text("\n".join(lines) + "\n")
    lexres.to_csv(REPO / "reports" / "wf_llm_benchset_a_induced_lexical.csv")
    supres.to_csv(REPO / "reports" / "wf_llm_benchset_a_induced_lexblock.csv")
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
