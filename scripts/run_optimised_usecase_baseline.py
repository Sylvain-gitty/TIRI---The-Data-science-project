"""What does a better-written use case do to the baseline model?

Rebuilds every brief-dependent feature from the optimised specs
(`scripts/optimise_usecases.py`) and re-runs `notebooks/main/06_baseline_logreg.ipynb`'s pipeline
on both arms, so the only thing that differs between them is **the text of six specs**.

Why this is cheap: the papers do not change
-------------------------------------------
Optimising a use case changes the *brief*, never the corpus. So the 8,704 paper-embedding columns
in `papers_fe.parquet` are reused untouched, and the only GPU work is embedding **6 briefs x 2
models = 12 short strings** — seconds, not a corpus pass. What is rebuilt:

  `cos_brief_{jasper,qwen4b}` + their ranks   re-embedded brief, cached paper vectors
  the 22 `lex_*` columns                      rebuilt on CPU by `build_lexical_features`
  everything else                             identical, by construction

`qwen8b` is left at its original values and excluded from both arms' feature sets, because only
jasper and qwen4b were asked for and re-embedding a third model would change what the arms have in
common rather than what differs.

The design constraint that makes the number mean anything
---------------------------------------------------------
The optimised specs were built **without any access to labels, papers, or the corpus** — every
token traces to another field of the same spec file (see `optimise_usecases.py`, and
`--provenance` there prints the mapping). That matters because the cold-start features match brief
text against paper text: a brief written by reading the relevant papers would fit the brief to the
labels, and the resulting "gain" would be manufactured. `DATA_BRIEF.md` honest-limit #2 records
that exact failure in the benchset corpus.

Evaluation, and it is notebook 06's, not a new one
--------------------------------------------------
`StratifiedGroupKFold` grouped by `first_author` and stratified on `use_case x label`: outer split 0
gives `final_holdout` (~20%), inner 5-fold within the remaining pool gives the validation folds.
`StandardScaler -> PCA(50, whiten)` on the embedding block and median-impute -> scale on the rest,
both fitted **inside** each split, then `LogisticRegression(class_weight="balanced")`.

⚠️ **This is a POOLED model and `CONTEXT.md` §1 says no pooled model ever ships** — a LogReg reads
`use_case_key` off the raw embedding at 96.2% accuracy, so a pooled gain can be use-case identity
rather than brief quality. It is reported because it is the baseline that was asked for, and the
**within-silo** table beside it is the production-relevant one.

⚠️ Prevalence here is **26-77% positive**, roughly 20x production. F2 and any threshold-derived
number are measured in the wrong regime and do not transfer (`CONTEXT.md` §3).

    python scripts/run_optimised_usecase_baseline.py
"""

from __future__ import annotations

import argparse
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

from ensemble_eval_utils import to_md  # noqa: E402
from lexical_features import build_lexical_features  # noqa: E402
from optimise_usecases import USE_CASES, _flat  # noqa: E402

FE = REPO / "data" / "processed" / "papers_fe.parquet"
PC = REPO / "data" / "processed" / "papers_combined.parquet"
SPEC_DIRS = {"optimised": REPO / "data" / "raw" / "optimised",
             "prose_only": REPO / "data" / "raw" / "optimised_prose_only"}
BRIEF_VECS_FMT = "data/processed/{arm}_brief_vectors.parquet"
OUT_MD = REPO / "reports" / "wf_optimised_usecase_baseline.md"
OUT_CSV = REPO / "reports" / "wf_optimised_usecase_baseline.csv"

EMB_MODELS = {"qwen4b": "Qwen/Qwen3-Embedding-4B",
              "jasper": "infgrad/Jasper-Token-Compression-600M"}
# embed_benchsets.USE_CASE_COLS - the fields the cosine brief is built from.
BRIEF_COLS = ["use_case_name", "problem_statement", "objective",
              "domain_industry", "domain_application"]

RANDOM_STATE, N_SPLITS_OUTER, N_SPLITS_INNER, PCA_N = 0, 5, 5, 50
GROUP_COL, USE_CASE_COL = "first_author", "use_case_key"
METADATA_COLS = ["year", "paper_age", "has_abstract", "n_authors", "citation_count"]


def cosine(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    v = vec / (np.linalg.norm(vec) + 1e-12)
    return m @ v


def pct_rank(v: np.ndarray) -> np.ndarray:
    order = v.argsort()
    r = np.empty(len(v), dtype=float)
    r[order] = np.arange(len(v), dtype=float)
    return r / max(len(v) - 1, 1)


def brief_text(row: pd.Series) -> str:
    return "\n".join(f"{c}: {row[c]}" for c in BRIEF_COLS
                     if str(row.get(c) or "").strip())


def embed_briefs(frames: dict[str, pd.Series], arm: str) -> pd.DataFrame:
    """One vector per (use case, model) for the optimised brief. 12 short strings, cached."""
    cache = REPO / BRIEF_VECS_FMT.format(arm=arm)
    if cache.exists():
        print(f"{arm} brief vectors: cached ({cache.name})")
        return pd.read_parquet(cache)
    from embedding_utils import MODEL_CONFIGS, embed_texts

    keys = sorted(frames)
    texts = [brief_text(frames[k]) for k in keys]
    print(f"embedding {len(texts)} optimised briefs x {len(EMB_MODELS)} models via Modal "
          f"(median {int(np.median([len(t) for t in texts]))} chars)")
    rows = []
    for short, model in EMB_MODELS.items():
        vecs = np.asarray(embed_texts(model, texts, prefix=MODEL_CONFIGS[model]["query_prefix"],
                                      is_query=True), dtype=np.float32)
        print(f"  {short}: {vecs.shape}")
        rows += [{"use_case_key": k, "model": short, "embedding": v}
                 for k, v in zip(keys, vecs)]
    out = pd.DataFrame(rows)
    out.to_parquet(cache, index=False)
    return out


def build_optimised_frame(fe: pd.DataFrame, arm: str) -> pd.DataFrame:
    """`fe` with cos_brief_* and lex_* recomputed from the optimised specs. Papers untouched."""
    d = SPEC_DIRS[arm]
    specs = {k: _flat(json.loads((d / f"{k}.usecase.json").read_text())) for k in USE_CASES}
    pc = pd.read_parquet(PC, columns=["paper_id", "use_case_key", "title", "abstract"])
    txt = fe[["paper_id", "use_case_key"]].merge(pc, on=["paper_id", "use_case_key"], how="left",
                                                 validate="one_to_one")
    for col in _flat(json.loads((d / f"{USE_CASES[0]}.usecase.json").read_text())):
        txt[col] = txt.use_case_key.map(lambda k: specs[k][col])

    out = fe.copy()

    # --- lexical block, rebuilt on the optimised brief ---
    block = build_lexical_features(txt)
    for c in block.columns:
        if f"lex_{c}" in out.columns:
            out[f"lex_{c}"] = block[c].to_numpy()

    # --- cosine-to-brief, re-embedded brief x cached paper vectors ---
    bv = embed_briefs({k: pd.Series(specs[k]) for k in USE_CASES}, arm) \
        .set_index(["model", "use_case_key"]).embedding
    for short in EMB_MODELS:
        cols = [c for c in fe.columns if c.startswith(f"emb_{short}_")]
        cos = np.full(len(out), np.nan)
        rnk = np.full(len(out), np.nan)
        for uc, g in out.groupby(USE_CASE_COL):
            idx = g.index.to_numpy()
            c = cosine(g[cols].to_numpy(np.float64), np.asarray(bv.loc[(short, uc)], np.float64))
            cos[idx], rnk[idx] = c, pct_rank(c)
        out[f"cos_brief_{short}"] = cos
        out[f"rank_cos_brief_{short}"] = rnk
    return out


COLD_FEATURES = ["cos_brief_jasper", "cos_brief_qwen4b",
                 "lex_bm25_obj", "lex_bm25_must", "lex_bm25_nice", "lex_overlap_must_frac"]


def cold_start(arms: dict) -> pd.DataFrame:
    """The zero-label rung: each brief-reading feature scored raw, per use case, no fitting.

    This is the arm that should move, and the fitted baseline is the arm that should not.
    `wf_spec_quality_answer.md` §2 measured the regimes inverting: prose damage costs a
    zero-label matcher 0.15-0.26 and a fitted one <=0.009. The baseline below is fitted on ~1,478
    labels, i.e. **24x** the 60-label budget at which brief quality had already stopped mattering.
    Reporting only that arm would answer a different question than the one asked.

    No fitting, no threshold, no split - every row is held out from a model that does not exist.
    """
    rows = []
    for feat in COLD_FEATURES:
        for uc in sorted(arms["baseline"][USE_CASE_COL].unique()):
            rec = {"feature": feat, "use_case": uc}
            ok = True
            for name, frame in arms.items():
                g = frame[frame[USE_CASE_COL] == uc]
                if g.y.nunique() < 2 or g[feat].isna().all():
                    ok = False
                    break
                rec[name] = roc_auc_score(g.y, g[feat].fillna(g[feat].median()))
            if ok:
                rows.append(rec)
    d = pd.DataFrame(rows)
    for name in arms:
        if name != "baseline":
            d[f"d_{name}"] = d[name] - d.baseline
    return d


def metrics(y, p, s) -> dict:
    return {
        "n": len(y), "pos_rate": float(np.mean(y)),
        "roc_auc": roc_auc_score(y, s), "avg_precision": average_precision_score(y, s),
        "f2": fbeta_score(y, p, beta=2, zero_division=0),
        "f1": fbeta_score(y, p, beta=1, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "pred_pos_rate": float(np.mean(p)),
    }


def pipeline(emb_cols: list[str], feat_cols: list[str]) -> Pipeline:
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("embedding_pca", Pipeline([("scale", StandardScaler()),
                                        ("pca", PCA(PCA_N, whiten=True,
                                                    random_state=RANDOM_STATE))]), emb_cols),
            ("non_embedding", Pipeline([("impute", SimpleImputer(strategy="median")),
                                        ("scale", StandardScaler())]), feat_cols),
        ])),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])


def evaluate(df: pd.DataFrame, emb_cols: list[str], feat_cols: list[str], arm: str) -> pd.DataFrame:
    strat = df[USE_CASE_COL].astype(str) + "__" + df.y.astype(str)
    outer = list(StratifiedGroupKFold(N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_STATE)
                 .split(df, strat, groups=df[GROUP_COL]))
    pool_idx, hold_idx = outer[0]
    pool = df.iloc[pool_idx].reset_index(drop=True)
    hold = df.iloc[hold_idx].reset_index(drop=True)

    strat_in = pool[USE_CASE_COL].astype(str) + "__" + pool.y.astype(str)
    inner = list(StratifiedGroupKFold(N_SPLITS_INNER, shuffle=True, random_state=RANDOM_STATE)
                 .split(pool, strat_in, groups=pool[GROUP_COL]))

    rows = []
    val_oof = np.full(len(pool), np.nan)
    for tr, va in inner:
        m = pipeline(emb_cols, feat_cols).fit(pool.iloc[tr][emb_cols + feat_cols],
                                              pool.iloc[tr].y)
        val_oof[va] = m.predict_proba(pool.iloc[va][emb_cols + feat_cols])[:, 1]

    final = pipeline(emb_cols, feat_cols).fit(pool[emb_cols + feat_cols], pool.y)
    tr_s = final.predict_proba(pool[emb_cols + feat_cols])[:, 1]
    ho_s = final.predict_proba(hold[emb_cols + feat_cols])[:, 1]

    for fold, y, s in (("train", pool.y.to_numpy(), tr_s),
                       ("validate", pool.y.to_numpy(), val_oof),
                       ("test", hold.y.to_numpy(), ho_s)):
        rows.append({"arm": arm, "fold": fold, **metrics(y, (s >= 0.5).astype(int), s)})

    # within-silo on the held-out fold - the production-relevant view (CONTEXT.md §1)
    for uc, g in hold.assign(score=ho_s).groupby(USE_CASE_COL):
        if g.y.nunique() < 2:
            continue
        rows.append({"arm": arm, "fold": f"test/{uc}",
                     **metrics(g.y.to_numpy(), (g.score >= 0.5).astype(int), g.score)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()

    fe = pd.read_parquet(FE)
    emb_cols = [c for c in fe.columns if c.startswith(("emb_jasper_", "emb_qwen4b_"))]
    lex_cols = [c for c in fe.columns if c.startswith("lex_")]
    feat_cols = lex_cols + ["cos_brief_jasper", "rank_cos_brief_jasper",
                            "cos_brief_qwen4b", "rank_cos_brief_qwen4b"] + METADATA_COLS
    print(f"{len(fe):,} rows x {fe[USE_CASE_COL].nunique()} use cases, {fe.y.mean():.1%} positive")
    print(f"{len(emb_cols):,} embedding cols (jasper+qwen4b), {len(feat_cols)} non-embedding")

    arms = {"baseline": fe,
            "optimised": build_optimised_frame(fe, "optimised"),
            "prose_only": build_optimised_frame(fe, "prose_only")}
    moved = {f"{c} [{a}]": float(np.corrcoef(fe[c].fillna(0), arms[a][c].fillna(0))[0, 1])
             for a in ("optimised", "prose_only")
             for c in ("cos_brief_qwen4b", "lex_bm25_obj", "lex_bm25_must")}
    print("\ninstrument check - correlation between the two arms' features "
          "(1.000 would mean the brief change did nothing):")
    for k, v in moved.items():
        print(f"  {k:22s} r={v:.4f}")

    cold = cold_start(arms)
    cold.to_csv(str(OUT_CSV).replace(".csv", "_coldstart.csv"), index=False)
    cs = cold.groupby("feature")[list(arms) + [f"d_{a}" for a in arms if a != "baseline"]].mean().round(4)
    for a in arms:
        if a != "baseline":
            cs[f"better_{a}"] = cold.groupby("feature")[f"d_{a}"].apply(
                lambda s: f"{int((s > 0).sum())}/{len(s)}")
    print("\n=== COLD START (0 labels, one feature, no fitting) - the regime that should move ===")
    print(to_md(cs))

    res = pd.concat([evaluate(f, emb_cols, feat_cols, a) for a, f in arms.items()],
                    ignore_index=True)
    res.to_csv(OUT_CSV, index=False)

    piv = res.pivot(index="fold", columns="arm")
    order = ["train", "validate", "test"] + sorted(f for f in res.fold.unique() if "/" in f)
    delta = pd.DataFrame({f"{m} [{a}]": piv[m][a] - piv[m]["baseline"]
                          for a in ("optimised", "prose_only")
                          for m in ["f2", "roc_auc", "recall"]}).loc[order].round(4)

    print("\n" + to_md(res.set_index(["arm", "fold"]).round(4)))
    print("\noptimised - baseline:\n" + to_md(delta))
    print(f"\nwrote {OUT_CSV}")

    OUT_MD.write_text(_report(res, delta, moved, fe, cs, cold), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def _report(res, delta, moved, fe, cs, cold) -> str:
    return (
        "# Do better-written use cases move the baseline model?\n\n"
        "Status: **measured.** `scripts/optimise_usecases.py` rewrote TIRI's six specs per "
        "[`wf_spec_quality_answer.md`](wf_spec_quality_answer.md); this re-ran "
        "`notebooks/main/06_baseline_logreg.ipynb`'s pipeline on the result. $0 beyond 12 short "
        "brief embeddings — the papers never change, so their vectors are reused untouched.\n\n"
        "🟢 clears the 0.03 noise floor · 🟡 real but under it · ⚪ engineering finding\n\n"
        "## 0. How to read any number here\n\n"
        "**The only difference between the two arms is the text of six spec files.** Same papers, "
        "same paper embeddings, same splits, same seeds, same pipeline. So any difference is "
        "attributable to the brief — which is exactly why the *provenance* of the rewrite matters "
        "more than the rewrite itself.\n\n"
        "🔴 **The rewrite never saw a label, a paper, or the corpus.** Every token in the optimised "
        "specs traces to another field of the same spec file — `performance_criteria`, "
        "`constraints`, `decision_criteria` and `notes` are all analyst-written and **read by "
        "neither** feature path. The optimisation moves the analyst's own words into the fields the "
        "model actually reads; it adds no knowledge. Had the briefs been written by reading the "
        "relevant papers, the cold-start features match brief text against paper text and the gain "
        "would be manufactured — `DATA_BRIEF.md` honest-limit #2 records that failure in the "
        "benchset corpus. Run `optimise_usecases.py --provenance` to audit the mapping.\n\n"
        "**Metrics.** `f2` weights recall 4x precision — the screening-appropriate one, since "
        "missing a relevant paper costs more than reading an irrelevant one. `roc_auc` is ranking "
        "quality (0.500 = coin flip). `avg_precision` is the precision-recall area, which is more "
        "informative than AUC when positives are rare. All at the model's own 0.5 threshold, "
        "nothing tuned.\n\n"
        "⚠️ **A gap under 0.03 is not established** (`CONTEXT.md` §5).\n\n"
        "⚠️ **This is a POOLED model, and `CONTEXT.md` §1 says no pooled model ever ships** — a "
        "LogReg reads `use_case_key` off the raw embedding at 96.2% accuracy, so a pooled gain can "
        "be use-case identity rather than brief quality. The per-use-case `test/*` rows are the "
        "production-relevant view.\n\n"
        f"⚠️ **Prevalence is {fe.y.mean():.1%} positive**, roughly 20x production. Every F2, "
        "precision and threshold number here is measured in the wrong regime and does not transfer "
        "(`CONTEXT.md` §3).\n\n"
        "**Instrument check.** Correlation between the two arms' brief-dependent features — 1.000 "
        "would mean the rewrite changed nothing and the comparison is empty:\n\n"
        + "\n".join(f"- `{k}` r={v:.4f}" for k, v in moved.items()) + "\n\n"
        "## 1. 🟢 Cold start — the regime this was supposed to move\n\n"
        "Each brief-reading feature scored **raw**, per use case: no labels, no fitting, no "
        "threshold. This is the rung a better spec is supposed to help, and it is the one the "
        "fitted baseline in §2 cannot see.\n\n"
        + to_md(cs) + "\n\n"
        "**Hard finding — the two halves of the rewrite pull in opposite directions, and the "
        "`prose_only` arm separates them cleanly.** Enriching the objective from the analyst's own "
        "unread fields lifts the qwen4b cosine by **+0.027 on average and 4 of 6 use cases, with "
        "all four gains clearing the 0.03 floor** (`solar_leo` +0.061, `ner` +0.047, "
        "`carbon_capture` +0.036, `soil_microbiome` +0.034). Padding the term lists costs "
        "**−0.028** on `lex_bm25_must`. **`prose_only` keeps the entire cosine gain and none of the "
        "loss** — every term-derived feature returns to exactly 0.000.\n\n"
        "### The term-list rule, v1 → v2, and where it stopped\n\n"
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
        "| **Never truncate** | capping `soil_microbiome`'s 10 hand-written terms at 8 cost −0.077 | "
        "\"5–8\" is a floor and a quality bar, **never a cap** |\n\n"
        "🟡 **One regression survives, and I am stopping rather than fixing it.** `ner` still loses "
        "**−0.116** from the single term \"News text processing\". The diagnosis is dilution rather "
        "than genericness: its tokens (`news` 0.16, `text` 0.40, `processing` 0.27 document "
        "frequency in that pool) are *not* more common than the existing ones (`entity` 0.78, "
        "`named` 0.61), but adding three moderately-common tokens to a five-token query raises "
        "scores across ~30–40% of the pool and swamps the rare discriminative ones "
        "(`disambiguation` 0.055, `linking` 0.109).\n\n"
        "🔴 **That points at a document-frequency filter on candidate terms — and I have not built "
        "it, deliberately.** This rule has now been iterated twice while watching the same "
        "six-use-case measurement. A third fix aimed at `ner` specifically would be tuning the rule "
        "on its own evaluation, which is the selection-on-holdout failure `CONTEXT.md` §4 exists to "
        "prevent. The df filter is recorded as a **pre-registered proposal** to test on the 28 "
        "benchset briefs, a surface not used for any of this — not applied here.\n\n"
        "**So the recommended configuration is `prose_only`**: it captures the entire measured gain "
        "(+0.027 cosine) with every term-derived feature at exactly 0.000. On this evidence, "
        "enrich the objective and **leave the analyst's term lists alone**.\n\n"
        "🔴 **The original framing of this section is kept below, because it was my mistake and "
        "the record should show it.** "
        "`wf_spec_quality_answer.md` §4 asks for \"5–8 **precise, discriminative** phrases\"; the "
        "rule I wrote padded to 8 with whatever `domain_technology_focus` and "
        "`performance_criteria[].metric` happened to contain. That is `keyword_flood` with domain "
        "words instead of generic ones, and it reproduced that failure mode almost exactly "
        "(−0.028 here against −0.031 measured on set B). Two specific mechanisms:\n\n"
        "1. **Metric names are terrible must-include terms.** `ner` lost **−0.118** because "
        "\"Precision\" was promoted from `performance_criteria` into `terms_must_include`, where it "
        "matches virtually every NLP paper ever written.\n"
        "2. **\"5–8\" must be a floor and a quality bar, never a cap.** `soil_microbiome` lost "
        "**−0.077** because its 10 hand-written, precise terms were **truncated to 8** to satisfy a "
        "recommendation derived from a corpus whose specs happened to carry 6–8.\n\n"
        "**So §4's MVP row needs one word changed and one exclusion added:** *at least* 5–8 "
        "discriminative phrases, and never promote a metric name into a term list.\n\n"
        "*ELI18: this asks \"if you had zero labelled papers and could only sort by how well each "
        "paper matches the brief, how good would that sort be?\" 0.500 is a coin flip. It is the "
        "situation every new project starts in, and it is the only situation where the words in the "
        "brief are all the model has.*\n\n"
        "## 2. Both arms, every fold — the fitted baseline\n\n" + to_md(res.set_index(["arm", "fold"]).round(4)) + "\n\n"
        "## 2. Optimised − baseline\n\n" + to_md(delta) + "\n"
    )


if __name__ == "__main__":
    main()
