"""
embedding_utils.py — shared embedding logic behind compare_embeddings.py and
future_work/train_baseline_classifier.py (parked, see that file), so both embed text
identically (same model registry, same prefixes, same title/abstract join) instead of
two copies quietly drifting apart.

ALSO home to the evaluation-metrics layer all three comparison scripts share (label
masks, cross-validated ROC-AUC, and the query-conditioned ranking metrics —
query_similarity_auc/recall@k/WSS@95) — grouped here rather than a new module because
they're built directly on top of the POSITIVE_VALUES/NEGATIVE_VALUES/PASS_VALUES label
handling this file already owns. See build_label_masks and evaluate_representation.

Nothing in here is specific to comparison OR training — it's "given a model name and
some papers, produce vectors", plus the export-loading and label-evaluation helpers
every caller needs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
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
    # Hosted models accessed via OpenRouter's OpenAI-compatible /embeddings endpoint
    # (needs OPENROUTER_API_KEY — see embed_via_openrouter below), evaluated in
    # notebooks/experiments/wf_embedding_model_bakeoff.ipynb against the local baseline above.
    # None of these have a documented asymmetric query/passage usage convention for
    # their OpenRouter-hosted form (unlike BAAI/bge-small-en-v1.5's local fastembed
    # build above), so prefixes are left empty rather than guessed — flagged in the
    # bake-off notebook, not assumed.
    "qwen/qwen3-embedding-8b": {"backend": "openrouter", "query_prefix": "", "passage_prefix": "", "title_abstract_sep": ". "},
    "openai/text-embedding-3-large": {"backend": "openrouter", "query_prefix": "", "passage_prefix": "", "title_abstract_sep": ". "},
    "baai/bge-m3": {"backend": "openrouter", "query_prefix": "", "passage_prefix": "", "title_abstract_sep": ". "},
    "mistralai/mistral-embed-2312": {"backend": "openrouter", "query_prefix": "", "passage_prefix": "", "title_abstract_sep": ". "},
    "google/gemini-embedding-2": {"backend": "openrouter", "query_prefix": "", "passage_prefix": "", "title_abstract_sep": ". "},
    "nvidia/nemotron-3-embed-1b:free": {"backend": "openrouter", "query_prefix": "", "passage_prefix": "", "title_abstract_sep": ". "},
    "perplexity/pplx-embed-v1-4b": {"backend": "openrouter", "query_prefix": "", "passage_prefix": "", "title_abstract_sep": ". "},
}

OPENROUTER_MODELS = [
    "qwen/qwen3-embedding-8b",
    "openai/text-embedding-3-large",
    "baai/bge-m3",
    "mistralai/mistral-embed-2312",
    "google/gemini-embedding-2",
    "nvidia/nemotron-3-embed-1b:free",
    "perplexity/pplx-embed-v1-4b",
]

# 4 heavier HF models run on GPU via the Modal app in scripts/modal_embeddings.py
# (too slow/impractical for local CPU) -- registered here with their real HF
# identifiers for consistent display, mapped to the short model_key that Modal app's
# EmbeddingWorker.embed() dispatches on internally (see that file's dispatch dict).
# None have a single string-prefix asymmetric convention (SPECTER2 switches adapters,
# QZhou/Qwen3-Embedding-4B use an instruction template, Jasper uses prompt_name= /
# compression_ratio=) -- all handled server-side in modal_embeddings.py based on the
# is_query flag threaded through _raw_embed, not via query_prefix/passage_prefix here.
MODAL_MODEL_KEYS = {
    "allenai/specter2": "specter2",
    "Kingsoft-LLM/QZhou-Embedding": "qzhou",
    "infgrad/Jasper-Token-Compression-600M": "jasper",
    "Qwen/Qwen3-Embedding-4B": "qwen3_4b",
}
MODAL_MODELS = list(MODAL_MODEL_KEYS.keys())
for _modal_name in MODAL_MODELS:
    MODEL_CONFIGS[_modal_name] = {
        "backend": "modal", "query_prefix": "", "passage_prefix": "",
        "title_abstract_sep": "[SEP]" if _modal_name == "allenai/specter2" else ". ",
    }
del _modal_name

# Fallback used for any model NOT in MODEL_CONFIGS (e.g. an ad hoc model name you type in
# without registering it here first): no prefixes, fastembed backend, plain ". " join.
_UNREGISTERED_MODEL_DEFAULT = {
    "backend": "fastembed", "query_prefix": "", "passage_prefix": "", "title_abstract_sep": ". ",
}

# Triage/review label values the agent's exports use.
POSITIVE_VALUES = {"positive"}
NEGATIVE_VALUES = {"negative"}
PASS_VALUES = {"pass"}
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


def build_label_masks(df: pd.DataFrame, label_col: str) -> dict:
    """Every boolean mask + integer label array the comparison scripts need, built once
    instead of copy-pasted (and silently drifting) across compare_embeddings.py,
    compare_ner_models.py, and compare_combined_features.py.

    Two label readings are built, not one — see reports/metrics_rework_and_rerun.md for
    why:
    - "strict": positive vs. negative only, `pass` and unlabelled rows excluded entirely.
      This is the reading every AUC number in this repo used before the rework — it's
      informative about the CLEAR-CUT cases, but a deployed triage tool can't skip the
      ambiguous middle the way this metric does, so it tends to read more optimistic than
      an analyst's real experience.
    - "conservative": positive vs. everything the analyst did NOT accept (negative AND
      pass). A harsher, deployment-shaped second reading of the exact same data — the gap
      between strict and conservative on any run is itself informative (a big gap means
      the `pass` rows are doing a lot of the classification work).
    """
    n = len(df)
    if label_col not in df.columns:
        empty_bool, empty_int = np.zeros(n, dtype=bool), np.zeros(n, dtype=int)
        return {
            "has_labels": False,
            "pos_mask": empty_bool, "neg_mask": empty_bool, "pass_mask": empty_bool,
            "strict_mask": empty_bool, "strict_y": empty_int,
            "conservative_mask": empty_bool, "conservative_y": empty_int,
            "n_pos": 0, "n_neg": 0, "n_pass": 0,
        }

    labels = df[label_col]
    pos_mask = labels.isin(POSITIVE_VALUES).to_numpy()
    neg_mask = labels.isin(NEGATIVE_VALUES).to_numpy()
    pass_mask = labels.isin(PASS_VALUES).to_numpy()

    return {
        "has_labels": True,
        "pos_mask": pos_mask, "neg_mask": neg_mask, "pass_mask": pass_mask,
        "strict_mask": pos_mask | neg_mask,
        "strict_y": pos_mask.astype(int),  # only meaningful where strict_mask is True
        "conservative_mask": pos_mask | neg_mask | pass_mask,
        "conservative_y": pos_mask.astype(int),  # only meaningful where conservative_mask is True
        "n_pos": int(pos_mask.sum()), "n_neg": int(neg_mask.sum()), "n_pass": int(pass_mask.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Embedding backends — fastembed neural models, sentence-transformers, TF-IDF+SVD,
# or OpenRouter's hosted /embeddings endpoint
# ─────────────────────────────────────────────────────────────────────────────

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


def embed_via_openrouter(
    model_name: str, texts: list[str], api_key: str | None = None,
    batch_size: int = 64, max_retries: int = 5,
) -> np.ndarray:
    """Embed `texts` through OpenRouter's OpenAI-compatible /embeddings endpoint,
    batching requests (the endpoint accepts `input` as a list of strings in one call)
    so 1000+ papers isn't 1000+ separate HTTP round trips.

    Raises on any non-2xx response after retries are exhausted (rather than silently
    returning zeros/NaNs) — an embedding model that isn't actually reachable is a
    finding worth surfacing loudly, not a row to quietly skip.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (env var or explicit api_key=).")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload = {"model": model_name, "input": batch, "encoding_format": "float"}

        for attempt in range(max_retries):
            resp = requests.post(OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                break
            retryable = resp.status_code == 429 or resp.status_code >= 500
            if not retryable or attempt == max_retries - 1:
                raise RuntimeError(
                    f"OpenRouter embeddings request failed for model={model_name!r} "
                    f"(status {resp.status_code}): {resp.text[:500]}"
                )
            wait = float(resp.headers.get("Retry-After", 2 ** attempt))
            time.sleep(wait)

        data = resp.json()["data"]
        # OpenRouter/OpenAI responses aren't guaranteed to preserve input order —
        # each item carries its own `index`, so sort back into request order rather
        # than assuming positional alignment.
        data.sort(key=lambda item: item["index"])
        all_vectors.extend(item["embedding"] for item in data)

    return np.array(all_vectors)


_modal_worker = None


def embed_via_modal(model_name: str, texts: list[str], is_query: bool = False) -> np.ndarray:
    """Embed via the GPU-backed Modal app in scripts/modal_embeddings.py (SPECTER2,
    QZhou-Embedding, Jasper-Token-Compression-600M, Qwen3-Embedding-4B). Deploy once
    with `modal deploy scripts/modal_embeddings.py`; this just looks up the already-
    deployed app by name (no redeploy needed per call, and the app's own EmbeddingWorker
    keeps whichever model was last used warm across calls)."""
    if model_name not in MODAL_MODEL_KEYS:
        raise ValueError(f"{model_name!r} is not a registered Modal-backed model (see MODAL_MODEL_KEYS)")
    global _modal_worker
    if _modal_worker is None:
        import modal
        _modal_worker = modal.Cls.from_name("tiri-embeddings", "EmbeddingWorker")()
    vectors = _modal_worker.embed.remote(MODAL_MODEL_KEYS[model_name], texts, is_query)
    return np.array(vectors)


def _raw_embed(model_name: str, backend: str, texts: list[str], is_query: bool = False) -> np.ndarray:
    """Run the actual model forward pass — no prefixing, no timing, just text in,
    vectors out."""
    if model_name == "tfidf":
        vectorizer = TfidfVectorizer(max_features=20_000, stop_words="english")
        sparse = vectorizer.fit_transform(texts)
        n_components = max(2, min(100, sparse.shape[0] - 1, sparse.shape[1] - 1))
        return TruncatedSVD(n_components=n_components, random_state=42).fit_transform(sparse)
    if backend == "modal":
        return embed_via_modal(model_name, texts, is_query=is_query)
    if backend == "openrouter":
        return embed_via_openrouter(model_name, texts)
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


def embed_texts(model_name: str, texts: list[str], prefix: str = "", is_query: bool = False) -> np.ndarray:
    """Embed arbitrary text — e.g. a use-case query built from several export columns
    rather than a paper's title+abstract — with model_name's configured prefix applied
    manually. For paper title+abstract text, use embed_papers/embed_corpus_and_use_case
    instead, which also handle the title/abstract join; this is for callers that already
    have a single finished string per item.

    `is_query` only matters for Modal-backed models (see MODAL_MODEL_KEYS) whose
    query/document asymmetry is handled server-side (adapter switch / prompt_name=),
    not via `prefix` — pass it whenever `texts` are queries, harmless no-op otherwise."""
    cfg = MODEL_CONFIGS.get(model_name, _UNREGISTERED_MODEL_DEFAULT)
    prefixed = [prefix + t for t in texts]
    return _raw_embed(model_name, cfg["backend"], prefixed, is_query=is_query)


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
# Evaluation metrics — see reports/metrics_rework_and_rerun.md for the full reasoning
# behind why each of these exists and what it does/doesn't measure.
# ─────────────────────────────────────────────────────────────────────────────

def cross_validated_roc_auc(X: np.ndarray, y: np.ndarray) -> dict:
    """Mean (+ std, + the raw per-fold list) cross-validated ROC-AUC for a fresh
    LogisticRegression, mirroring academic_research_agent's model.py:
    _cross_validated_roc_auc.

    WHAT THIS DOES NOT MEASURE: the classifier is fit on paper vectors alone — the
    use-case/query vector never enters this computation. A representation can score well
    here (label-separable in the abstract) while being useless for ranking-by-similarity-
    to-a-specific-query, and vice versa. See query_similarity_auc below for the
    query-conditioned counterpart; report both, don't treat this one as "the" score.

    fold_aucs is returned (not just the mean) so callers doing a cross-representation
    comparison can run a PAIRED test across folds instead of eyeballing a point
    difference — StratifiedKFold(shuffle=False) makes fold membership depend only on `y`,
    so two representations scored on the identical y get identical fold row-membership,
    making per-fold AUCs directly pairable.
    """
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    k = min(MAX_CV_FOLDS, n_pos, n_neg)
    if k < 2:
        return {"roc_auc": None, "roc_auc_std": None, "n_folds": 0, "fold_aucs": []}

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
        return {"roc_auc": None, "roc_auc_std": None, "n_folds": 0, "fold_aucs": []}
    return {
        "roc_auc": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "n_folds": len(aucs),
        "fold_aucs": aucs,
    }


def query_similarity_scores(paper_vectors: np.ndarray, use_case_vector: np.ndarray) -> np.ndarray:
    """Cosine similarity between every paper vector and the use-case query vector — the
    actual ranking signal a nearest-neighbour retrieval step (and the agent's own triage
    ranking) uses. Deliberately re-derived here rather than imported from
    latent_space_utils.normalize_rows, to keep this module's only dependency on that one
    a documentation-level one (see query_similarity_auc's docstring), not an import edge."""
    norms = np.linalg.norm(paper_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V_norm = paper_vectors / norms
    q_norm = use_case_vector / (np.linalg.norm(use_case_vector) or 1.0)
    return V_norm @ q_norm


def retrieval_ranking_metrics(
    paper_vectors: np.ndarray, use_case_vector: np.ndarray, y: np.ndarray,
    recall_fractions: tuple[float, ...] = (0.1, 0.2), wss_target_recall: float = 0.95,
) -> dict:
    """Score ranking-by-similarity-to-the-use-case-query against the labels — the
    query-conditioned counterpart cross_validated_roc_auc is missing (that one ignores
    the query; this one ignores nothing, and needs no classifier/training at all, since
    it's just a ranking of raw cosine similarities).

    query_similarity_auc: AUC of "papers ranked by cosine-to-use-case" vs. the binary
      label. Same 0.5=chance/1.0=perfect scale as roc_auc, directly comparable to it, but
      answers a different question: does ranking BY THIS QUERY separate relevant from
      not, rather than "is this space linearly separable by label at all, however scored".
    recall_at_Xpct: of the top X% of papers by that ranking, what fraction of the true
      positives an analyst would already have seen. Maps directly onto "how much of the
      ranked list do I have to read".
    wss_at_95: Work Saved over Sampling at 95% recall (Cohen et al. 2006) — a standard
      citation-screening-automation metric: the fraction of the corpus an analyst could
      skip while still catching 95% of the positives, minus the 5% they'd have saved by
      chance alone. 0 = no better than random ordering; higher is better.
    """
    n = len(y)
    n_pos = int(y.sum())
    empty = {
        "query_similarity_auc": None,
        **{f"recall_at_{int(f * 100)}pct": None for f in recall_fractions},
        "wss_at_95": None, "wss_at_95_recall_achieved": None, "n": n, "n_pos": n_pos,
    }
    if n_pos == 0 or n_pos == n:
        return empty

    scores = query_similarity_scores(paper_vectors, use_case_vector)
    auc = float(roc_auc_score(y, scores))

    order = np.argsort(-scores)  # descending: most-similar-to-query first
    cum_pos = np.cumsum(y[order])

    recalls = {}
    for frac in recall_fractions:
        k = max(1, int(round(frac * n)))
        recalls[f"recall_at_{int(frac * 100)}pct"] = float(cum_pos[k - 1] / n_pos)

    target_count = int(np.ceil(wss_target_recall * n_pos))
    cutoff_idx = int(np.searchsorted(cum_pos, target_count, side="left"))
    screened = cutoff_idx + 1
    recall_achieved = float(cum_pos[cutoff_idx] / n_pos)
    wss = float((n - screened) / n - (1 - wss_target_recall))

    return {
        "query_similarity_auc": auc,
        **recalls,
        "wss_at_95": wss,
        "wss_at_95_recall_achieved": recall_achieved,
        "n": n,
        "n_pos": n_pos,
    }


def evaluate_representation(paper_vectors: np.ndarray, use_case_vector: np.ndarray, masks: dict) -> dict:
    """The full "test this representation against the labelled data" bundle every
    comparison script runs: classifier separability (roc_auc, query-blind) AND
    query-conditioned ranking (query_similarity_auc/recall/wss, query-aware) — each under
    BOTH the strict and conservative label readings from build_label_masks. Flat dict,
    ready to fold straight into a results row.
    """
    out = {}
    if not masks["has_labels"]:
        for mode in ("strict", "conservative"):
            out.update({
                f"n_{mode}": 0, f"roc_auc_{mode}": None, f"roc_auc_{mode}_std": None,
                f"n_folds_{mode}": 0, f"query_auc_{mode}": None,
                f"recall_at_10pct_{mode}": None, f"recall_at_20pct_{mode}": None,
                f"wss_at_95_{mode}": None,
            })
        return out

    for mode in ("strict", "conservative"):
        mask = masks[f"{mode}_mask"]
        y_full = masks[f"{mode}_y"]
        n_labelled = int(mask.sum())
        out[f"n_{mode}"] = n_labelled

        if n_labelled < 6 or len(set(y_full[mask].tolist())) < 2:
            out.update({
                f"roc_auc_{mode}": None, f"roc_auc_{mode}_std": None, f"n_folds_{mode}": 0,
                f"query_auc_{mode}": None, f"recall_at_10pct_{mode}": None,
                f"recall_at_20pct_{mode}": None, f"wss_at_95_{mode}": None,
            })
            continue

        y = y_full[mask]
        roc = cross_validated_roc_auc(paper_vectors[mask], y)
        ranking = retrieval_ranking_metrics(paper_vectors[mask], use_case_vector, y)

        out[f"roc_auc_{mode}"] = roc["roc_auc"]
        out[f"roc_auc_{mode}_std"] = roc["roc_auc_std"]
        out[f"n_folds_{mode}"] = roc["n_folds"]
        out[f"query_auc_{mode}"] = ranking["query_similarity_auc"]
        out[f"recall_at_10pct_{mode}"] = ranking["recall_at_10pct"]
        out[f"recall_at_20pct_{mode}"] = ranking["recall_at_20pct"]
        out[f"wss_at_95_{mode}"] = ranking["wss_at_95"]

    return out


def round_floats(d: dict, ndigits: int = 4) -> dict:
    """Round every float value in a flat dict for CSV/console output; leave None, ints,
    and non-numeric values untouched. Shared so the three comparison scripts' result-row
    construction doesn't each re-implement the same None-guarded rounding."""
    return {k: (round(v, ndigits) if isinstance(v, float) else v) for k, v in d.items()}


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
