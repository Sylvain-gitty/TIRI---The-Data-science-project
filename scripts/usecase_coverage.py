"""usecase_coverage.py — which use-case themes would actually diversify the training set.

THE JOB
-------
`notebooks/experiments/wf_usecase_diversity.ipynb` measured the use cases we HAVE. This
answers the next question: given that map, which use cases should we go and acquire?

It scores a catalogue of candidate themes (`scripts/usecase_themes.json`, editable — that
is the point) against every brief already cached, on two axes that pull in opposite
directions:

- **novelty** — distance to the nearest existing use case. High is good: it is new ground.
- **domain fit** — closeness to an explicit `domain_anchor` brief. High is good: it is
  still the kind of work we screen for.

Maximising either alone is useless. Pure novelty finds you a use case about medieval
poetry; pure domain fit finds you a second copy of `cement_binders`. The output is the
band in between, plus a greedy portfolio that spreads picks across *different* gaps rather
than crowding into the largest one.

RE-RUNNING AS DATA ARRIVES — the design constraint
--------------------------------------------------
"Existing use cases" is not a hardcoded list. It is whatever briefs are in
`data/processed/embeddings_cache/*_usecases.parquet` at run time, which is exactly what
`embed_benchsets.py` and the feature-engineering notebook write. Onboard a new use case,
let its brief get cached, re-run this, and every gap and every verdict updates. Candidate
vectors are cached too, so a re-run after adding three new themes embeds three texts.

WHAT THIS TOOL IS ALLOWED TO CLAIM — read before believing an output
--------------------------------------------------------------------
A candidate theme has a brief and NO CORPUS. So every score here is brief-space geometry
standing in for where that use case's papers would land. That substitution was measured,
not assumed: across the 34 use cases that have both, brief-to-brief similarity predicts
corpus-to-corpus similarity at **Spearman 0.60** (0.60 on Jasper, 0.60 on Qwen3-4B, 561
pairs). Real, and nowhere near deterministic.

The honest reading of any row below is therefore *"this theme probably lands about here"*,
useful for choosing what to go and label, and not a substitute for measuring the corpus
once it exists. Two consequences worth stating plainly:

- A theme flagged `duplicate-risk` is a **prompt to check**, not a verdict. Confirm against
  the real corpus before discarding a candidate.
- The novelty ranking is reliable at the level of "clearly new" vs "clearly covered". Do
  not read a 0.02 difference between two adjacent rows as meaningful — that is well inside
  what a 0.60 rank correlation can reorder.

CALIBRATION, NOT MAGIC NUMBERS
------------------------------
Every threshold is derived from your own data at run time and printed in the report, so
none of them has to be trusted on faith and all of them move as the data changes:

- the **overlap tiers** are the 75th and 90th **percentiles** of how close your existing use
  cases already sit to *their own* nearest neighbours. A candidate is flagged when it would
  land closer to something you have than most of your use cases sit to each other.
- the **domain floor** is the midpoint between your industrial and biomedical use cases —
  a guard rail that rejects an off-topic theme, not a ranking. See the long comment in
  `coverage_table` for why it is not a percentile.

Novelty is thresholded on **percentile ranks, never on raw cosines**, and that is the single
most important implementation decision here. Two earlier versions got it wrong the same way:
first a hardcoded 0.70 duplicate line carried over from the diversity notebook's *corpus*-
space centroids (a pure scale error — the known-leaking `hall_2012`/`radjenovic_2013` pair
scores 0.871 there and 0.711 in brief space), then run-time thresholds that were still
absolute cosines. That second version agreed with itself on only **45%** of themes between
Jasper and Qwen3-4B despite the two models ranking the themes almost identically (Spearman
0.84 on novelty). On ranks, verdict agreement is **91%**. Diversity notebook §5 is the
general statement of this: read orderings, not cosines.

WHICH OUTPUT TO TRUST — measured, and they are not equal
--------------------------------------------------------
Re-running under a second embedding model is the check, and it puts a clear ordering on the
three outputs this tool produces:

| Output | Jasper vs Qwen3-4B | Use it for |
|---|---|---|
| **Sector ranking** | Spearman **0.97** | The decision. Which areas are open ground. |
| **Per-theme verdict** | **91%** identical | A shortlist inside a chosen sector. |
| **Ordered portfolio** | **4/12** shared | Illustration only — see below. |

The portfolio is a greedy farthest-point walk, so each pick changes the scores of every
remaining candidate and small differences cascade. Two models proposing different 12-item
lists is that chaos, not disagreement about the underlying geometry — they still agree on
which *sectors* those items come from. **Read the sector table, then choose themes within a
sector for reasons this tool cannot see** (data availability, customer demand, labelling
cost). Do not hand someone the numbered list as a procurement plan.

Usage:
    python scripts/usecase_coverage.py                    # full run, Jasper, top 10
    python scripts/usecase_coverage.py --model qwen4b     # second opinion
    python scripts/usecase_coverage.py --top 15
    python scripts/usecase_coverage.py --dry-run          # what would be embedded, embed nothing
    python scripts/usecase_coverage.py --themes my.json   # a different candidate catalogue
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import usecase_diversity_utils as udu  # noqa: E402
from embed_benchsets import USE_CASE_COLS  # noqa: E402
from embedding_utils import MODEL_CONFIGS, embed_texts  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "processed" / "embeddings_cache"
REPORTS = REPO / "reports"
DEFAULT_THEMES = Path(__file__).resolve().parent / "usecase_themes.json"

# Percentiles of the existing nearest-neighbour distribution that set the two overlap
# tiers. Percentiles, not fixed cosines, because the absolute scale depends on the model and
# on which use cases exist — see this module's docstring on the scale error that motivated
# this. p90 = "closer than 90% of your use cases sit to their own nearest neighbour".
OVERLAP_TIERS = {"duplicate-risk": 90, "overlaps-existing": 75}


def brief_text(record: dict) -> str:
    """The five brief fields, joined exactly as embed_benchsets.py joins them.

    Any divergence here — a different field order, a different separator — puts candidate
    vectors in a subtly different space from the cached real briefs and every distance in
    the report becomes meaningless. Hence USE_CASE_COLS is imported, never retyped."""
    parts = []
    for column in USE_CASE_COLS:
        value = record.get(column)
        if isinstance(value, (list, tuple)):
            value = " ; ".join(str(v) for v in value if str(v).strip())
        if value is not None and str(value).strip():
            parts.append(str(value))
    return " ".join(parts)


def load_catalogue(path: Path, anchor: str | None = None) -> dict:
    """The catalogue, with one named anchor selected into `domain_anchor`.

    Anchors are the lens: the same 79 candidate themes rank differently depending on
    whether you are asking "is this heavy industry" or "is this decarbonisation". Switching
    lens changes only the domain axis — novelty against your existing use cases is a
    property of the geometry and does not move."""
    catalogue = json.loads(path.read_text())
    for required in ("anchors", "themes"):
        if required not in catalogue:
            raise ValueError(f"{path}: catalogue is missing {required!r}")
    name = anchor or catalogue.get("default_anchor") or next(iter(catalogue["anchors"]))
    if name not in catalogue["anchors"]:
        raise ValueError(f"unknown anchor {name!r}; have {sorted(catalogue['anchors'])}")
    selected = dict(catalogue["anchors"][name])
    catalogue["anchor_name"] = name
    catalogue["in_domain_reference"] = selected.pop("in_domain_reference")
    catalogue["domain_anchor"] = selected
    keys = [t["key"] for t in catalogue["themes"]]
    if len(set(keys)) != len(keys):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"{path}: duplicate theme keys {dupes}")
    return catalogue


def theme_vectors(model_key: str, catalogue: dict, dry_run: bool = False) -> dict[str, np.ndarray]:
    """key -> brief vector for every candidate theme plus `__anchor__`, embedding only what
    is missing. Same cache shape as the real `*_usecases.parquet` files, kept in a separate
    file so a candidate can never be mistaken for a real use case by another reader."""
    model_name = udu.MODELS[model_key]["name"]
    path = CACHE / f"themes_{udu.safe_name(model_name)}_usecases.parquet"

    cached: dict[str, np.ndarray] = {}
    if path.exists():
        frame = pd.read_parquet(path)
        cached = {k: np.asarray(v, dtype=np.float32)
                  for k, v in zip(frame["use_case_key"], frame["embedding"])}

    wanted = {f"__anchor__{catalogue['anchor_name']}": catalogue["domain_anchor"]}
    wanted.update({t["key"]: t for t in catalogue["themes"]})
    missing = {k: v for k, v in wanted.items() if k not in cached}

    if missing and dry_run:
        print(f"  {model_key}: WOULD EMBED {len(missing)} theme brief(s)")
        return cached
    if missing:
        print(f"  {model_key}: embedding {len(missing)} new theme brief(s) ...")
        cfg = MODEL_CONFIGS[model_name]
        texts = [brief_text(v) for v in missing.values()]
        vectors = embed_texts(model_name, texts, prefix=cfg["query_prefix"], is_query=True)
        cached.update({k: np.asarray(v, dtype=np.float32) for k, v in zip(missing, vectors)})
        pd.DataFrame({"use_case_key": list(cached), "embedding": list(cached.values())}
                     ).to_parquet(path, index=False)
    else:
        print(f"  {model_key}: all {len(wanted)} theme briefs already cached")
    return cached


def existing_briefs(model_key: str) -> dict[str, np.ndarray]:
    """Every real use case with a cached brief in this model — the live corpus and, where
    the model has them, the benchset collections. This is what makes the tool re-runnable:
    onboard a use case, cache its brief, and it joins the comparison set automatically."""
    briefs: dict[str, np.ndarray] = {}
    for surface in udu.MODELS[model_key]["surfaces"]:
        briefs.update(udu.load_brief_vectors(model_key, surface))
    return briefs


def coverage_table(model_key: str, catalogue: dict, dry_run: bool = False) -> tuple[pd.DataFrame, dict]:
    """One row per candidate theme, plus the calibration landmarks the report prints."""
    existing = existing_briefs(model_key)
    themes = theme_vectors(model_key, catalogue, dry_run=dry_run)
    if dry_run:
        return pd.DataFrame(), {}
    anchor = themes.pop(f"__anchor__{catalogue['anchor_name']}")
    themes = {k: v for k, v in themes.items() if not k.startswith("__anchor__")}

    exist_keys = sorted(existing)
    E = np.vstack([existing[k] for k in exist_keys]).astype(np.float64)
    # Centre on the existing briefs, matching the diversity notebook's convention: raw
    # cosine is dominated by the direction all academic prose shares (see that notebook §2).
    centre = E.mean(axis=0, keepdims=True)
    E = E - centre

    cand_keys = [t["key"] for t in catalogue["themes"] if t["key"] in themes]
    C = np.vstack([themes[k] for k in cand_keys]).astype(np.float64) - centre
    a = (np.asarray(anchor, dtype=np.float64)[None, :] - centre)

    sim_ce = udu.cosine_matrix(C, E)                 # candidate x existing
    fit_c = udu.cosine_matrix(C, a).ravel()          # candidate -> anchor
    fit_e = pd.Series(udu.cosine_matrix(E, a).ravel(), index=exist_keys)

    # Pass mark: the weakest use case you already treat as in-scope.
    reference = [k for k in catalogue["in_domain_reference"] if k in fit_e.index]
    if not reference:
        raise ValueError(f"none of in_domain_reference {catalogue['in_domain_reference']} "
                         f"has a cached brief for {model_key}")
    pass_mark = float(fit_e[reference].min())

    sim_ee = udu.cosine_matrix(E, E)
    np.fill_diagonal(sim_ee, -np.inf)
    existing_nn = sim_ee.max(axis=1)

    # Both axes are converted to PERCENTILE RANKS against your existing use cases before any
    # threshold is applied, and this is load-bearing rather than cosmetic. Absolute cosines
    # are not comparable across embedding models — the diversity notebook §5 found the models
    # agree on the *ordering* of close pairs while disagreeing on the values. Thresholding
    # raw cosines made this tool agree with itself on only 45% of themes between Jasper and
    # Qwen3-4B despite rank correlations of 0.84 (novelty) and 0.81 (domain fit). Ranks are
    # invariant to that rescaling, so the verdicts inherit the agreement rather than the noise.
    domain_pct = np.array([(fit_e.to_numpy() < f).mean() for f in fit_c])
    overlap_pct = np.array([(existing_nn < c).mean() for c in sim_ce.max(axis=1)])

    # DOMAIN FIT IS A GUARD RAIL, NOT A RANKING. Measured on the shipped catalogue, this axis
    # cleanly separates industrial use cases (fit 0.13 to 0.34) from the benchset's biomedical
    # ones (-0.14 to ~0) and resolves almost nothing *within* industrial: all 58 candidates
    # land above the 90th percentile of a comparison set that is 82% biomedical.
    #
    # So the threshold is the midpoint between the two modes, not a percentile of the pooled
    # distribution. Requiring a candidate to out-score `carbon_capture` on "industrial-ness"
    # put 21 plainly-industrial themes below the bar on one model and 3 on the other — a
    # verdict flapping on noise at a saturated ceiling, not a real distinction. The midpoint
    # rejects a genuinely off-topic theme and passes everything in the industrial mode, which
    # is the entire job this axis can honestly do.
    in_mode, out_mode = fit_e[fit_e >= pass_mark], fit_e[fit_e < pass_mark]
    domain_floor = float((in_mode.mean() + out_mode.mean()) / 2) if len(out_mode) else pass_mark

    sector = {t["key"]: t.get("sector", "") for t in catalogue["themes"]}
    name = {t["key"]: t["use_case_name"] for t in catalogue["themes"]}
    table = pd.DataFrame({
        "theme": cand_keys,
        "sector": [sector[k] for k in cand_keys],
        "use_case_name": [name[k] for k in cand_keys],
        "nearest_existing": [exist_keys[i] for i in sim_ce.argmax(axis=1)],
        "nearest_cos": sim_ce.max(axis=1),
        "overlap_pct": overlap_pct,
        "domain_fit": fit_c,
        "domain_pct": domain_pct,
    })
    tiers = {label: p / 100.0 for label, p in OVERLAP_TIERS.items()}
    table["verdict"] = np.where(
        table.domain_fit < domain_floor, "off-domain",
        np.where(table.overlap_pct >= tiers["duplicate-risk"], "duplicate-risk",
                 np.where(table.overlap_pct >= tiers["overlaps-existing"],
                          "overlaps-existing", "adopt")))

    landmarks = {
        "model": model_key,
        "anchor": catalogue["anchor_name"],
        "n_existing": len(exist_keys),
        "n_candidates": len(cand_keys),
        "pass_mark": pass_mark,
        "pass_mark_set_by": fit_e[reference].idxmin(),
        "domain_floor": domain_floor,
        "existing_nn_median": float(np.median(existing_nn)),
        "existing_nn_max": float(existing_nn.max()),
        "existing_nn_max_pair": " / ".join(sorted(
            {exist_keys[int(existing_nn.argmax())],
             exist_keys[int(sim_ee[int(existing_nn.argmax())].argmax())]})),
        "tiers": tiers,
        "existing_fit": fit_e.sort_values(ascending=False),
        "sim_ce": pd.DataFrame(sim_ce, index=cand_keys, columns=exist_keys),
        "C": C, "E": E, "exist_keys": exist_keys, "cand_keys": cand_keys,
    }
    return table.sort_values("nearest_cos").reset_index(drop=True), landmarks


def select_portfolio(table: pd.DataFrame, landmarks: dict, k: int = 10) -> pd.DataFrame:
    """Greedy farthest-point selection over the eligible candidates.

    Picking the k most individually-novel themes would pile them into whichever single
    region of the space is emptiest. This instead adds, at each step, the candidate whose
    nearest neighbour in *everything chosen so far* (existing use cases plus earlier picks)
    is furthest away — the standard k-centre greedy. The result spreads across distinct
    gaps, which is what diversifying a training set actually needs.

    `min_cos_to_set` in the output is the pick's similarity to its closest neighbour at the
    moment it was chosen: read it as how much genuinely new ground that pick adds."""
    eligible = table[table.verdict == "adopt"].copy()
    if eligible.empty:
        return eligible.assign(pick=[], min_cos_to_set=[])

    sim_ce = landmarks["sim_ce"].loc[eligible.theme]
    C = pd.DataFrame(landmarks["C"], index=landmarks["cand_keys"]).loc[eligible.theme].to_numpy()
    sim_cc = udu.cosine_matrix(C, C)

    closest = sim_ce.to_numpy().max(axis=1)   # to the existing set
    chosen: list[int] = []
    rows = []
    for pick in range(1, min(k, len(eligible)) + 1):
        i = int(np.argmin(closest))
        rows.append(dict(pick=pick, theme=eligible.theme.iloc[i], min_cos_to_set=float(closest[i])))
        chosen.append(i)
        closest = np.maximum(closest, sim_cc[:, i])
        closest[chosen] = np.inf                # never pick the same theme twice

    return (pd.DataFrame(rows).merge(eligible, on="theme")
            [["pick", "theme", "sector", "use_case_name", "min_cos_to_set",
              "nearest_existing", "nearest_cos", "domain_fit"]])


def write_report(table: pd.DataFrame, portfolio: pd.DataFrame, landmarks: dict, path: Path) -> None:
    counts = table.verdict.value_counts().to_dict()
    lines = [
        f"# Use-case coverage — where new data would diversify the training set",
        "",
        f"Generated {date.today()} by `scripts/usecase_coverage.py` on `{landmarks['model']}`, "
        f"under the **{landmarks['anchor']}** anchor, against **{landmarks['n_existing']} existing use cases** and "
        f"**{landmarks['n_candidates']} candidate themes** from `scripts/usecase_themes.json`.",
        "",
        "> **Candidate themes have briefs, not corpora.** Every score here is brief-space "
        "geometry standing in for where a use case's papers would land. Measured across the "
        "34 use cases that have both, that substitution holds at Spearman 0.60 — real, but "
        "loose. Read this as a shortlist to go and acquire, not as a measurement. A "
        "`duplicate-risk` flag is a prompt to check against the real corpus, not a verdict.",
        "",
        "## Calibration (computed at run time, not hardcoded)",
        "",
        "| Landmark | Value | What it means |",
        "|---|---|---|",
        f"| Domain pass mark | {landmarks['pass_mark']:.3f} | Weakest `in_domain_reference` use "
        f"case (`{landmarks['pass_mark_set_by']}`). |",
        f"| Domain floor (guard rail) | {landmarks['domain_floor']:.3f} | Midpoint between your "
        "industrial and biomedical use cases. Below it is `off-domain`. This axis is a guard "
        "rail, not a ranking — see the stability note below. |",
        f"| Existing nearest-neighbour, median | {landmarks['existing_nn_median']:.3f} | How "
        "close a typical pair of use cases you already have sits. |",
        f"| Existing nearest-neighbour, max | {landmarks['existing_nn_max']:.3f} | The closest "
        f"real pair (`{landmarks['existing_nn_max_pair']}`) — the only pair ever measured to "
        "inflate a LOGO score. Nothing here should be allowed to reach it. |",
        "| `overlaps-existing` line | p75 | Closer to something you own than 75% of your use "
        "cases are to their own nearest neighbour. |",
        "| `duplicate-risk` line | p90 | Same at 90%. Check against the real corpus before "
        "acquiring. |",
        "",
        f"**Verdicts:** " + ", ".join(f"{v} {c}" for v, c in sorted(counts.items())),
        "",
        "### Where your existing use cases sit against the domain anchor",
        "",
        "The guard rail is the **domain floor**, not the pass mark: the two "
        "`in_domain_reference` use cases only serve to locate the industrial mode, and "
        "everything in that mode passes. Use cases sitting between the floor and the pass "
        "mark are in-domain but less central than your reference pair — a fact about the "
        "current portfolio, not a problem.",
        "",
        f"| Use case | domain_fit | vs guard rail ({landmarks['domain_floor']:.3f}) |",
        "|---|---|---|",
    ] + [
        f"| `{k}` | {v:.3f} | {'in domain' if v >= landmarks['domain_floor'] else 'off-domain'} |"
        for k, v in landmarks["existing_fit"].items()
        if v >= landmarks["existing_fit"].quantile(0.72)
    ] + [
        f"| *(remaining {int((landmarks['existing_fit'] < landmarks['existing_fit'].quantile(0.72)).sum())} "
        "— all benchset biomedical/psychology collections)* | "
        f"{landmarks['existing_fit'].min():.3f} to "
        f"{landmarks['existing_fit'][landmarks['existing_fit'] < landmarks['existing_fit'].quantile(0.72)].max():.3f}"
        " | below |",
        "",
    ]

    rollup = (table.assign(open=lambda d: d.verdict == "adopt")
              .groupby("sector")
              .agg(candidates=("theme", "size"), open_ground=("open", "sum"),
                   median_overlap_pct=("overlap_pct", "median"),
                   median_domain_fit=("domain_fit", "median"))
              .sort_values("median_overlap_pct").round(3))
    lines += [
        "## Which sectors are open ground — **this is the output to act on**",
        "",
        "`median_overlap_pct` is how close the sector's themes typically sit to the nearest "
        "thing you already have — **lower means more new ground**. This ranking is the most "
        "stable thing the tool produces: Jasper and Qwen3-4B agree on it at Spearman **0.97**, "
        "against 91% agreement on individual verdicts and only 4/12 on the ordered portfolio "
        "below. Decide the *sector* here; pick themes within it for reasons this tool cannot "
        "see (data availability, customer demand, labelling cost).",
        "",
        rollup.to_markdown(), "",
    ]

    show = portfolio.assign(min_cos_to_set=lambda d: d.min_cos_to_set.round(3),
                            nearest_cos=lambda d: d.nearest_cos.round(3),
                            domain_fit=lambda d: d.domain_fit.round(3))
    lines += [
        f"## An illustrative portfolio (top {len(portfolio)}) — **not a procurement plan**",
        "",
        "Greedy farthest-point selection: each pick is the candidate furthest from everything "
        "chosen so far, so the set spreads across *different* gaps instead of crowding into "
        "the largest one. `min_cos_to_set` is the pick's similarity to its closest neighbour "
        "at the moment it was chosen — lower means more new ground.",
        "",
        "**Read this as one valid way to cover the open sectors, not as a ranking.** Greedy "
        "selection is chaotic: each pick reshuffles the remaining scores, so the two embedding "
        "models produce lists sharing only 4 of 12 entries while still agreeing on which "
        "sectors those entries come from.",
        "",
        show.to_markdown(index=False), "",
        "## Every candidate, by sector", ""]
    for sector, grp in table.groupby("sector", sort=True):
        lines += [f"### {sector}", ""]
        g = grp.sort_values("nearest_cos")[
            ["theme", "use_case_name", "verdict", "nearest_existing", "nearest_cos", "domain_fit"]]
        lines += [g.round(3).to_markdown(index=False), ""]

    lines += [
        "## How to re-run this",
        "",
        "```bash",
        "python scripts/usecase_coverage.py",
        "```",
        "",
        "The comparison set is whatever briefs are cached at run time, so onboarding a use "
        "case and re-running updates every gap and verdict automatically. Edit "
        "`scripts/usecase_themes.json` to add themes you are actually considering or delete "
        "ones you never would — only the new ones get embedded. Run `--model qwen4b` for a "
        "second opinion: the diversity notebook §5 found embedding models agree on the "
        "*ordering* of close pairs while disagreeing on absolute values, so a theme that "
        "changes verdict between models is one whose margin was never real.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def plot_coverage(table: pd.DataFrame, portfolio: pd.DataFrame, landmarks: dict, path: Path) -> None:
    """Sector ranking, with every theme shown as a point on its sector's row.

    Deliberately NOT the novelty-vs-domain-fit scatter this started as. On real data the
    domain-fit axis saturates — all 58 candidates land above the 88th percentile because
    they are all heavy industry by construction and the comparison set is 82% biomedical —
    so a 2D chart spent its whole vertical axis on a constant and collided its labels in a
    strip at the top. One informative axis, drawn as magnitude, beats two axes where one is
    noise. The strip overlay keeps the per-theme detail the bars average away."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = {"adopt": "#2a78d6", "overlaps-existing": "#eda100",
              "duplicate-risk": "#e34948", "off-domain": "#9b9a95"}
    order = (table.groupby("sector").overlap_pct.median().sort_values().index.tolist())
    picks = set(portfolio.theme)

    fig, ax = plt.subplots(figsize=(12, 6.6))
    for i, sector in enumerate(order):
        grp = table[table.sector == sector]
        ax.barh(i, grp.overlap_pct.median(), height=0.62, color="#cde2fb", zorder=2)
        jitter = np.linspace(-0.17, 0.17, len(grp)) if len(grp) > 1 else [0.0]
        for off, r in zip(jitter, grp.itertuples()):
            ax.scatter(r.overlap_pct, i + off, s=95 if r.theme in picks else 46,
                       color=colour.get(r.verdict, "#9b9a95"), zorder=4,
                       edgecolors="#0d366b" if r.theme in picks else "white",
                       linewidths=1.6 if r.theme in picks else 0.9)
        n_open = int((grp.verdict == "adopt").sum())
        ax.annotate(f"{n_open}/{len(grp)} open", (1.005, i), fontsize=7.5, color="#52514e",
                    va="center", annotation_clip=False)

    for label, x in landmarks["tiers"].items():
        ax.axvline(x, color="#e34948" if "duplicate" in label else "#eda100",
                   linestyle=":", linewidth=1.4, zorder=3)
        ax.annotate(label, (x, -0.55), fontsize=7.5, rotation=90,
                    color="#e34948" if "duplicate" in label else "#c98500",
                    ha="left", va="top", xytext=(3, 0), textcoords="offset points")

    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.invert_yaxis()
    ax.set(xlabel="overlap with what you already have  (bar = sector median, dots = themes)  "
                  "→  less new ground",
           title=f"Where more coverage would diversify the set — {landmarks['model']}  "
                 f"[{landmarks['anchor']} anchor]  ({landmarks['n_candidates']} themes "
                 f"vs {landmarks['n_existing']} existing)")
    ax.annotate("most open ground", (0.01, -0.42), fontsize=8.5, color="#184f95",
                fontweight="bold", va="center")
    ax.grid(axis="x", color="#dcdbd6", linewidth=0.6); ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=c, label=v)
               for v, c in colour.items() if v in set(table.verdict)]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                              markerfacecolor="none", markeredgecolor="#0d366b",
                              label="in illustrative portfolio"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="jasper", choices=udu.BOTH_SURFACE_MODELS,
                    help="embedding model; must have briefs cached on both surfaces")
    ap.add_argument("--themes", type=Path, default=DEFAULT_THEMES)
    ap.add_argument("--anchor", default=None,
                    help="which lens to score domain fit against (see `anchors` in the catalogue)")
    ap.add_argument("--top", type=int, default=10, help="portfolio size")
    ap.add_argument("--dry-run", action="store_true", help="report what would be embedded")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    catalogue = load_catalogue(args.themes, anchor=args.anchor)
    print(f"catalogue: {len(catalogue['themes'])} themes from {args.themes}"
          f"  |  anchor: {catalogue['anchor_name']}")

    table, landmarks = coverage_table(args.model, catalogue, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    portfolio = select_portfolio(table, landmarks, k=args.top)
    out = args.out or REPORTS / f"wf_usecase_coverage_{catalogue['anchor_name']}_{args.model}.md"
    write_report(table, portfolio, landmarks, out)
    figure = out.with_suffix(".png")
    plot_coverage(table, portfolio, landmarks, figure)

    print(f"\nverdicts: {table.verdict.value_counts().to_dict()}")
    print(f"domain pass mark {landmarks['pass_mark']:.3f} "
          f"(set by {landmarks['pass_mark_set_by']})\n")
    print(portfolio[["pick", "theme", "sector", "min_cos_to_set", "nearest_existing"]]
          .round(3).to_string(index=False))
    print(f"\nwrote {out}\nwrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
