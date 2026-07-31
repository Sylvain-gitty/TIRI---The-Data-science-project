"""
embedding_utils.py — shared embedding logic behind compare_embeddings.py AND
train_baseline_classifier.py, so both scripts embed text identically (same model
registry, same prefixes, same title/abstract join) instead of two copies quietly
drifting apart.

Nothing in here is specific to comparison OR training — it's "given a model name and
some papers, produce vectors", plus the export-loading helpers both scripts need.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# All registered model configs, keyed by the exact string fastembed/sentence-transformers
# expect. Each entry:
#   backend:        "fastembed" (local ONNX, no PyTorch) or "sentence-transformers"
#                    (local, but pulls in PyTorch — needed for models fastembed
#                    doesn't carry, e.g. Specter).
#   query_prefix:   glued in front of a QUERY text before embedding (e.g. the use case).
#   passage_prefix: glued in front of every paper's title+abstract before embedding.
#                   Both empty for symmetric models — see compare_embeddings.py's
#                   docstring PREFIXES section for why asymmetric models need these.
#   title_abstract_sep: how a paper's title and abstract are joined BEFORE any prefix.
#                   Matters because Specter was trained on "title[SEP]abstract" (its own
#                   model card's documented usage), not free-form prose.
MODEL_CONFIGS = {
    "sentence-transformers/allenai-specter": {
        "backend": "sentence-transformers",
        "query_prefix": "",
        "passage_prefix": "",
        "title_abstract_sep": "[SEP]",
    },
    "BAAI/bge-small-en-v1.5": {
        "backend": "fastembed",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "title_abstract_sep": ". ",
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "backend": "fastembed",
        "query_prefix": "",
        "passage_prefix": "",
        "title_abstract_sep": ". ",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "backend": "fastembed",
        "query_prefix": "",
        "passage_prefix": "",
        "title_abstract_sep": ". ",
    },
    # Registered but not run by default anywhere: on a 100-paper test corpus this took
    # ~795s to embed vs. 4-85s for every model above — fastembed's ONNX graph for this
    # model is dramatically slower than the rest on CPU. Still usable via --model(s).
    "nomic-ai/nomic-embed-text-v1.5": {
        "backend": "fastembed",
        "query_prefix": "search_query: ",
        "passage_prefix": "search_document: ",
        "title_abstract_sep": ". ",
    },
}

# Fallback used for any model NOT in MODEL_CONFIGS (e.g. an ad hoc model name you type in
# without registering it here first): no prefixes, fastembed backend, plain ". " join.
_UNREGISTERED_MODEL_DEFAULT = {
    "backend": "fastembed", "query_prefix": "", "passage_prefix": "", "title_abstract_sep": ". ",
}

# Triage/review label values the agent's exports use.
POSITIVE_VALUES = {"positive"}
NEGATIVE_VALUES = {"negative"}
MAX_CV_FOLDS = 5


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_export(path: Path) -> pd.DataFrame:
    """Load an academic_research_agent export (.parquet or .csv), no filtering."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (expected .parquet or .csv)")


def get_title_abstract(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return the raw (titles, abstracts) lists, one per paper. Kept as two separate
    lists (rather than pre-joined into one string) because HOW they're joined is a
    per-model decision — see MODEL_CONFIGS' title_abstract_sep and build_paper_texts."""
    titles = df.get("title", pd.Series([""] * len(df))).fillna("").tolist()
    abstracts = df.get("abstract", pd.Series([""] * len(df))).fillna("").tolist()
    return titles, abstracts


def drop_empty_rows(df: pd.DataFrame, titles: list[str], abstracts: list[str]):
    """Drop rows with no title AND no abstract text, from df and the two text lists in
    lockstep. Returns (df, titles, abstracts), reindexed. Shared so both scripts filter
    identically before embedding."""
    non_empty = [i for i, (t, a) in enumerate(zip(titles, abstracts)) if t.strip() or a.strip()]
    if len(non_empty) == len(titles):
        return df, titles, abstracts
    dropped = len(titles) - len(non_empty)
    print(f"  Dropping {dropped} rows with no title/abstract text.")
    df = df.iloc[non_empty].reset_index(drop=True)
    return df, [titles[i] for i in non_empty], [abstracts[i] for i in non_empty]


def build_paper_texts(titles: list[str], abstracts: list[str], sep: str) -> list[str]:
    """Join each paper's title+abstract with the separator THIS model expects. Default
    ". " matches academic_research_agent's own embeddings.paper_embedding_text; Specter
    gets "[SEP]" instead, matching its documented training format (see MODEL_CONFIGS).

    Deliberately NOT `.strip(sep)` — str.strip() treats its argument as a set of
    characters to strip, not a literal substring, so stripping "[SEP]" would eat any
    leading/trailing S/E/P characters off real titles/abstracts. Missing-side handling
    is done explicitly instead.
    """
    texts = []
    for title, abstract in zip(titles, abstracts):
        title, abstract = title.strip(), abstract.strip()
        texts.append(f"{title}{sep}{abstract}" if title and abstract else (title or abstract))
    return texts


def get_use_case_text(df: pd.DataFrame, use_case_col: str, override: str | None) -> str:
    """Return the text to embed as "the use case". An explicit override always wins;
    otherwise fall back to the export's use_case column (a short NAME, not the agent's
    full objective/key-terms JSON — the export doesn't carry that)."""
    if override:
        return override
    if use_case_col in df.columns:
        values = df[use_case_col].dropna().unique()
        if len(values) > 0:
            return str(values[0])
    raise ValueError(
        f"No use case text found: column '{use_case_col}' is missing/empty and "
        "no override text was given."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Embedding backends — fastembed neural models, sentence-transformers, or TF-IDF+SVD
# ─────────────────────────────────────────────────────────────────────────────

def _raw_embed(model_name: str, backend: str, texts: list[str]) -> np.ndarray:
    """Run the actual model forward pass — no prefixing, no timing, just text in,
    vectors out."""
    if model_name == "tfidf":
        vectorizer = TfidfVectorizer(max_features=20_000, stop_words="english")
        sparse = vectorizer.fit_transform(texts)
        n_components = max(2, min(100, sparse.shape[0] - 1, sparse.shape[1] - 1))
        return TruncatedSVD(n_components=n_components, random_state=42).fit_transform(sparse)
    if backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return np.asarray(model.encode(texts, show_progress_bar=False))
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=model_name)
    return np.array([v for v in model.embed(texts)])


def embed_papers(model_name: str, titles: list[str], abstracts: list[str]) -> tuple[np.ndarray, float]:
    """Embed every paper's title+abstract (joined + prefixed per MODEL_CONFIGS) with NO
    accompanying query — for callers that only need paper vectors (e.g. training a
    classifier on them), not a use-case comparison point.

    Returns (paper_vectors, seconds_elapsed).
    """
    cfg = MODEL_CONFIGS.get(model_name, _UNREGISTERED_MODEL_DEFAULT)
    paper_texts = build_paper_texts(titles, abstracts, cfg["title_abstract_sep"])
    passage_texts = [cfg["passage_prefix"] + t for t in paper_texts]

    start = time.monotonic()
    vectors = _raw_embed(model_name, cfg["backend"], passage_texts)
    elapsed = time.monotonic() - start
    return vectors, elapsed


def embed_corpus_and_use_case(
    model_name: str, titles: list[str], abstracts: list[str], use_case_text: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Like embed_papers, but ALSO embeds a use-case query (with query_prefix applied)
    through the same model call, so both land in the same vector space in one pass.

    Returns (paper_vectors, use_case_vector, seconds_elapsed).
    """
    cfg = MODEL_CONFIGS.get(model_name, _UNREGISTERED_MODEL_DEFAULT)
    paper_texts = build_paper_texts(titles, abstracts, cfg["title_abstract_sep"])
    passage_texts = [cfg["passage_prefix"] + t for t in paper_texts]
    query_text = cfg["query_prefix"] + use_case_text

    start = time.monotonic()
    vectors = _raw_embed(model_name, cfg["backend"], passage_texts + [query_text])
    elapsed = time.monotonic() - start

    return vectors[:-1], vectors[-1], elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Reusing an export's OWN precomputed embeddings, when they match the requested model
# ─────────────────────────────────────────────────────────────────────────────

def resolve_paper_vectors(
    df: pd.DataFrame, titles: list[str], abstracts: list[str], model_name: str
) -> tuple[np.ndarray, float, str]:
    """Get paper vectors for `model_name` — reusing the export's own precomputed
    `embedding` column when it was made with this exact model (matched on the model name
    only, ignoring the "@fastembed-x.y.z" backend/version suffix academic_research_agent
    stamps onto embed_model — see that repo's embeddings.embedding_model_key), and
    falling back to a fresh embed otherwise.

    Why bother: an academic_research_agent export already carries a vector per paper for
    whichever model built its triage ranking. If that's the SAME model this script was
    asked to use, re-embedding from scratch would just reproduce (approximately — a
    different fastembed version could shift pooling) vectors that are already sitting in
    the file, for no benefit — slower, and technically a different generation of the
    vectors than the ones the agent's own UI actually ranked by.

    Returns (paper_vectors, seconds_elapsed, source_note) — source_note says which path
    was taken, so callers can report it honestly rather than silently.
    """
    embed_model_col = df.get("embed_model")
    embedding_col = df.get("embedding")
    if embed_model_col is not None and embedding_col is not None:
        base_names = embed_model_col.dropna().apply(lambda s: s.split("@")[0]).unique()
        if len(base_names) == 1 and base_names[0] == model_name and embedding_col.notna().all():
            vectors = np.stack(embedding_col.to_numpy())
            return vectors, 0.0, f"reused precomputed embeddings from the export (embed_model matches {model_name})"

    vectors, elapsed = embed_papers(model_name, titles, abstracts)
    return vectors, elapsed, "freshly embedded (no matching precomputed column in the export)"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated ROC-AUC — the "does this space separate my labels" honesty check
# ─────────────────────────────────────────────────────────────────────────────

def cross_validated_roc_auc(X: np.ndarray, y: np.ndarray) -> dict:
    """Mean cross-validated ROC-AUC for a fresh LogisticRegression, mirroring
    academic_research_agent's model.py:_cross_validated_roc_auc exactly."""
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    k = min(MAX_CV_FOLDS, n_pos, n_neg)
    if k < 2:
        return {"roc_auc": None, "n_folds": 0}

    folds = StratifiedKFold(n_splits=k, shuffle=False)
    aucs = []
    for train_idx, test_idx in folds.split(X, y):
        y_test = y[test_idx]
        if len(set(y_test.tolist())) < 2:
            continue
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        pos_col = list(clf.classes_).index(1)
        y_score = clf.predict_proba(X[test_idx])[:, pos_col]
        aucs.append(roc_auc_score(y_test, y_score))

    if not aucs:
        return {"roc_auc": None, "n_folds": 0}
    return {"roc_auc": float(np.mean(aucs)), "n_folds": len(aucs)}


def cross_validated_oof_proba(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Out-of-fold predicted P(positive) for every labelled row, via the SAME
    StratifiedKFold + balanced LogisticRegression as cross_validated_roc_auc — but
    returning per-row probabilities instead of one aggregate score, so a caller can build
    a confusion matrix / classification report on genuinely held-out predictions (never a
    row scored by a model that trained on it).

    Rows caught in a degenerate fold (see cross_validated_roc_auc) get NaN — that can
    only happen for k < 2, which callers should already be guarding against.
    """
    proba = np.full(len(y), np.nan)
    folds = StratifiedKFold(n_splits=k, shuffle=False)
    for train_idx, test_idx in folds.split(X, y):
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        pos_col = list(clf.classes_).index(1)
        proba[test_idx] = clf.predict_proba(X[test_idx])[:, pos_col]
    return proba
