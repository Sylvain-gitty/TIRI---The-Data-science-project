"""Tier 1b — lexical query-conditioned features.

WHAT THIS IS FOR
----------------
Every feature here answers "how well does this paper match *this use case's brief*",
never "what is this paper about". That distinction is the whole point. A paper-only
feature (embedding dimension, citation count, venue) describes the paper, so a model
trained on it learns topic — which is exactly what does not survive a move to a new
use case, because relevance is a property of the (brief, paper) pair, not of the paper.

Measured, on the six current use cases, leave-one-use-case-out, plain LogisticRegression:
the 384-d paper embedding transfers at ~0.53 ROC-AUC while a handful of brief-relative
scalars reach ~0.64. The embedding is not noisy — it is a near-perfect use-case
fingerprint (a 6-way classifier reads `use_case_key` off it at 96% accuracy). See
reports/wf_featureengineering_review.md for the negative results that led here; the
one feature of eleven that worked (term overlap) was also the only one that read
the brief.

Under the siloed-per-customer architecture this is a *label-efficiency* claim rather
than a transfer claim: each customer trains their own weights, so the embedding never
has to transfer. What these features buy is a model that already knows what the
customer is looking for, and therefore needs fewer labels to become useful.

THE CONTROL THAT MAKES THIS FALSIFIABLE
---------------------------------------
Features like these can look good for a boring reason: if they secretly measure
abstract length or lexical richness, they would score well against *any* brief. So
every builder here takes a `brief_map` — a use-case -> use-case mapping saying whose
brief to score each paper against. Pass a derangement and the features are computed
against deliberately wrong briefs. If performance survives that, the features are not
reading the brief and the whole block should be thrown away. Run it before trusting
any number from this module (scripts/run_tier1b_control.py).

WHAT IS AND IS NOT FITTED
-------------------------
Nothing here is fitted on labels, so the whole block is safe to materialise into a
feature table. BM25 IDF and the percentile ranks are computed *within each use case's
own pool*: pool composition is known at scoring time in production, so this is
realistic rather than leaky, and it makes a feature mean the same thing in every silo.
Anything label-derived (PCA, scalers, the Tier 2 learned axis) belongs in a per-fold
Pipeline, not in here.

NULL IS NOT 0 (see CONTRIBUTING.md §1)
-----------------------------
`terms_exclude` is empty for three of the six current use cases. "No exclusion terms
were specified" and "no exclusion terms matched" are different facts, so the exclusion
columns come back NaN in the first case and 0.0 in the second, with `has_exclude_terms`
carrying the distinction. Callers must impute inside a fold, never here.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

# Okapi BM25 constants. The literature defaults; nothing here is tuned, and they are
# stated as constants rather than parameters because a per-use-case BM25 tuning knob is
# exactly the kind of thing that would overfit six use cases.
BM25_K1 = 1.5
BM25_B = 0.75

# Brief fields, in the order they are emitted. `dom` is deliberately included even
# though it is expected to be near-useless on its own: everything in a pool is on-domain
# (domain words are what retrieved it), so `dom` acts as the *baseline* that the Tier 1a
# contrast features subtract. Keeping it here means the same six keys exist in both tiers.
BRIEF_KEYS = ("obj", "prob", "must", "nice", "dom")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"\s+")


def normalise(text: str | None) -> str:
    """Lowercase, turn hyphens/slashes into spaces, collapse whitespace.

    Hyphen folding is not cosmetic: the briefs contain `alkali-activated` and
    `transformer-based NER` while papers write both hyphenated and spaced forms. Folding
    both sides to spaces makes a phrase match on either. Same reason `/` is folded.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return _WS_RE.sub(" ", str(text).lower().replace("-", " ").replace("/", " ")).strip()


def tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalise(text))


def _as_list(value) -> list[str]:
    """Term-list columns arrive as list, ndarray, None or NaN depending on the export."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if isinstance(value, float) and np.isnan(value):
        return []
    return [str(value)] if str(value).strip() else []


def build_brief_texts(row: pd.Series) -> dict[str, str]:
    """The five brief field texts for one use case, normalised and ready to tokenise."""
    return {
        "obj": normalise(row.get("objective")),
        "prob": normalise(row.get("problem_statement")),
        "must": normalise(" ; ".join(_as_list(row.get("terms_must_include")))),
        "nice": normalise(" ; ".join(_as_list(row.get("terms_nice_to_have")))),
        "dom": normalise(
            " ".join(
                [
                    str(row.get("domain_industry") or ""),
                    str(row.get("domain_application") or ""),
                    " ".join(_as_list(row.get("domain_technology_focus"))),
                ]
            )
        ),
    }


def collect_briefs(df: pd.DataFrame, use_case_col: str = "use_case_key") -> dict[str, dict]:
    """use_case_key -> {texts: {key: str}, terms: {must/nice/exclude: [str]}}.

    One row per use case is enough: the brief columns are broadcast across every paper
    in a pool by notebooks/main/01_data_compile.ipynb.
    """
    briefs: dict[str, dict] = {}
    for uc, group in df.groupby(use_case_col):
        row = group.iloc[0]
        briefs[uc] = {
            "texts": build_brief_texts(row),
            "terms": {
                "must": _as_list(row.get("terms_must_include")),
                "nice": _as_list(row.get("terms_nice_to_have")),
                "exclude": _as_list(row.get("terms_exclude")),
            },
        }
    return briefs


def phrase_pattern(term: str) -> re.Pattern:
    """Word-boundary regex for a possibly multi-word term, tolerant of whitespace runs.

    Plain `in` matching would count `NER` inside `generic`, and would miss
    `alkali activated` when the brief says `alkali-activated`. Both sides are normalised
    first, so this only has to handle the whitespace-run case.
    """
    parts = [re.escape(p) for p in normalise(term).split() if p]
    if not parts:
        return re.compile(r"(?!x)x")  # never matches
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b")


def _bm25_pool(paper_tokens: list[list[str]]) -> tuple[dict[str, float], float]:
    """IDF and average document length for one pool.

    Computed per use-case pool, not corpus-wide: use cases are siloed per customer, so
    a pool is the only corpus that exists at scoring time, and per-pool IDF makes a BM25
    score mean the same thing across silos of different sizes.
    """
    n_docs = len(paper_tokens)
    doc_freq: Counter = Counter()
    for tokens in paper_tokens:
        doc_freq.update(set(tokens))
    idf = {
        term: float(np.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)))
        for term, freq in doc_freq.items()
    }
    lengths = [len(t) for t in paper_tokens]
    return idf, (float(np.mean(lengths)) if lengths else 0.0)


def _bm25_scores(
    paper_tokens: list[list[str]], query_tokens: list[str], idf: dict[str, float], avgdl: float
) -> np.ndarray:
    """Okapi BM25 of every paper in a pool against one query."""
    if not query_tokens or avgdl == 0:
        return np.zeros(len(paper_tokens), dtype=float)
    query_terms = set(query_tokens)
    out = np.zeros(len(paper_tokens), dtype=float)
    for i, tokens in enumerate(paper_tokens):
        if not tokens:
            continue
        counts = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if freq == 0:
                continue
            denom = freq + BM25_K1 * (1.0 - BM25_B + BM25_B * doc_len / avgdl)
            score += idf.get(term, 0.0) * freq * (BM25_K1 + 1.0) / denom
        out[i] = score
    return out


def _pct_rank(values: np.ndarray) -> np.ndarray:
    """Percentile rank within the pool, NaN-safe.

    This is what lets a feature mean the same thing in every silo: raw BM25 scale depends
    on pool size and vocabulary, rank does not. Transductive (it reads the whole pool)
    but label-free, and production has the whole pool at scoring time.
    """
    out = np.full(len(values), np.nan, dtype=float)
    ok = ~np.isnan(values)
    if ok.sum() == 0:
        return out
    ranked = pd.Series(values[ok]).rank(method="average", pct=True).to_numpy()
    out[ok] = ranked
    return out


def build_lexical_features(
    df: pd.DataFrame,
    brief_map: dict[str, str] | None = None,
    use_case_col: str = "use_case_key",
    title_col: str = "title",
    abstract_col: str = "abstract",
) -> pd.DataFrame:
    """Tier 1b feature block, indexed like `df`.

    brief_map: use_case_key -> the use_case_key whose brief to score against. Defaults to
    identity. Pass a derangement to run the shuffled-brief control — the falsification
    test for this entire module (see the module docstring).
    """
    briefs = collect_briefs(df, use_case_col)
    brief_map = brief_map or {uc: uc for uc in briefs}

    paper_text = (
        df[title_col].fillna("").astype(str) + ". " + df[abstract_col].fillna("").astype(str)
    )
    normalised = paper_text.map(normalise)
    tokens = normalised.map(lambda t: _TOKEN_RE.findall(t))

    frame = pd.DataFrame(index=df.index)
    frame["n_tokens"] = tokens.map(len).astype(float)

    for column in (
        [f"bm25_{k}" for k in BRIEF_KEYS]
        + [f"rank_bm25_{k}" for k in BRIEF_KEYS]
        + [
            "overlap_must_n", "overlap_must_frac", "rank_overlap_must",
            "overlap_nice_n", "overlap_nice_frac", "rank_overlap_nice",
            "overlap_excl_n", "overlap_excl_frac",
            "overlap_must_per_1k", "overlap_nice_per_1k",
        ]
    ):
        frame[column] = np.nan
    frame["has_exclude_terms"] = False

    for uc, group in df.groupby(use_case_col):
        rows = group.index
        pool_tokens = [tokens.loc[i] for i in rows]
        idf, avgdl = _bm25_pool(pool_tokens)

        brief = briefs[brief_map[uc]]  # <- the control's single point of intervention

        for key in BRIEF_KEYS:
            scores = _bm25_scores(pool_tokens, tokenise(brief["texts"][key]), idf, avgdl)
            frame.loc[rows, f"bm25_{key}"] = scores
            frame.loc[rows, f"rank_bm25_{key}"] = _pct_rank(scores)

        pool_text = [normalised.loc[i] for i in rows]
        n_tokens = frame.loc[rows, "n_tokens"].to_numpy()

        for name, terms in (("must", brief["terms"]["must"]), ("nice", brief["terms"]["nice"])):
            if terms:
                patterns = [phrase_pattern(t) for t in terms]
                counts = np.array(
                    [sum(1 for p in patterns if p.search(text)) for text in pool_text], dtype=float
                )
                frac = counts / len(terms)
                frame.loc[rows, f"overlap_{name}_n"] = counts
                frame.loc[rows, f"overlap_{name}_frac"] = frac
                frame.loc[rows, f"rank_overlap_{name}"] = _pct_rank(frac)
                # Length control: the open question in reports/wf_featureengineering_review.md
                # §8 is whether term overlap is partly an abstract-length proxy. Emitting the
                # per-1k form alongside the raw count, with n_tokens, is what settles it.
                with np.errstate(divide="ignore", invalid="ignore"):
                    frame.loc[rows, f"overlap_{name}_per_1k"] = np.where(
                        n_tokens > 0, counts / (n_tokens / 1000.0), np.nan
                    )
            # else: leave NaN. An empty term list is "not specified", not "nothing matched".

        exclude = brief["terms"]["exclude"]
        if exclude:
            patterns = [phrase_pattern(t) for t in exclude]
            counts = np.array(
                [sum(1 for p in patterns if p.search(text)) for text in pool_text], dtype=float
            )
            frame.loc[rows, "overlap_excl_n"] = counts
            frame.loc[rows, "overlap_excl_frac"] = counts / len(exclude)
            frame.loc[rows, "has_exclude_terms"] = True

    frame["has_exclude_terms"] = frame["has_exclude_terms"].astype(bool)
    return frame


def derangements(keys: list[str], seed: int = 0) -> dict[str, str]:
    """A brief_map where no use case keeps its own brief.

    Sampled rather than enumerated so the control can be averaged over several wrong
    pairings instead of resting on one arbitrary shuffle.
    """
    rng = np.random.default_rng(seed)
    keys = list(keys)
    while True:
        shuffled = list(rng.permutation(keys))
        if all(a != b for a, b in zip(keys, shuffled)):
            return dict(zip(keys, shuffled))
