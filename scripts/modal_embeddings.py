"""modal_embeddings.py — GPU-backed embedding inference for 4 HF models too heavy
for local CPU (SPECTER2 via adapters, QZhou-Embedding, Jasper-Token-Compression-600M,
Qwen3-Embedding-4B). Called from notebooks/experiments/wf_embedding_model_bakeoff.ipynb via
scripts/embedding_utils.py's embed_via_modal, which looks up the deployed app by name
rather than needing this file present/redeployed on every call.

Deploy once (or after any edit here):
    modal deploy scripts/modal_embeddings.py

Quick standalone smoke test without touching the notebook:
    modal run scripts/modal_embeddings.py

Each model loads lazily on first use per container and is kept in memory for
`scaledown_window` seconds, so a smoke test immediately followed by the full embed
reuses the same warm container instead of reloading weights. HF checkpoint downloads
are cached on a persistent Modal Volume, so even a cold container restart skips
re-downloading multi-GB weights from the Hub.
"""

import os

import modal

APP_NAME = "tiri-embeddings"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers>=4.51.0",
    "sentence-transformers>=3.0.0",
    "adapters",
    "accelerate",
    "einops",
    "sentencepiece",
    "protobuf",
)

hf_cache_vol = modal.Volume.from_name("tiri-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

app = modal.App(APP_NAME, image=image)

# Same instruction template for every model here that wants one — QZhou and
# Qwen3-Embedding-4B both document this exact "Instruct: ...\nQuery:" convention;
# documents get no prefix on either. Jasper/SPECTER2 use their own encode-time
# mechanisms instead (prompt_name= / adapter switch), not a string prefix.
QUERY_INSTRUCTION = "Instruct: Given a research objective, retrieve relevant academic papers\nQuery:"


@app.cls(
    gpu="L4",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=3600,
    scaledown_window=300,
)
class EmbeddingWorker:
    @modal.enter()
    def setup(self):
        self.loaded = {}

    # ---- SPECTER2: adapters library, CLS pooling, asymmetric proximity/adhoc_query ----
    def _embed_specter2(self, texts, is_query):
        import torch
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        if "specter2" not in self.loaded:
            tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
            model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
            model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=False)
            model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query", set_active=False)
            model = model.to("cuda").eval()
            self.loaded["specter2"] = (tokenizer, model)

        tokenizer, model = self.loaded["specter2"]
        model.set_active_adapters("adhoc_query" if is_query else "proximity")

        out = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = model(**inputs)
            out.extend(output.last_hidden_state[:, 0, :].cpu().float().numpy().tolist())
        return out

    # ---- QZhou-Embedding: Qwen2.5-7B base, trust_remote_code, mean pooling, left pad ----
    def _embed_qzhou(self, texts, is_query):
        if "qzhou" not in self.loaded:
            from sentence_transformers import SentenceTransformer

            # QZhou's own modeling_qzhou.py (trust_remote_code) calls
            # DynamicCache.get_usable_length(), which newer transformers versions
            # removed in favor of get_seq_length() -- shim it back rather than
            # pinning an older transformers that would break Qwen3-Embedding-4B's
            # own (newer) model support in the same environment.
            from transformers.cache_utils import DynamicCache

            if not hasattr(DynamicCache, "get_usable_length"):
                DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)

            self.loaded["qzhou"] = SentenceTransformer(
                "Kingsoft-LLM/QZhou-Embedding",
                model_kwargs={"device_map": "cuda", "trust_remote_code": True, "torch_dtype": "bfloat16"},
                tokenizer_kwargs={"padding_side": "left", "trust_remote_code": True},
            )
        model = self.loaded["qzhou"]
        texts_in = [QUERY_INSTRUCTION + t for t in texts] if is_query else list(texts)
        return model.encode(texts_in, batch_size=16, show_progress_bar=False).tolist()

    # ---- Jasper-Token-Compression-600M: Qwen3-Embedding-0.6B + token compression ----
    def _embed_jasper(self, texts, is_query):
        if "jasper" not in self.loaded:
            from sentence_transformers import SentenceTransformer

            self.loaded["jasper"] = SentenceTransformer(
                "infgrad/Jasper-Token-Compression-600M", trust_remote_code=True, device="cuda"
            )
        model = self.loaded["jasper"]
        kwargs = {"compression_ratio": 0.5, "batch_size": 32, "show_progress_bar": False}
        if is_query:
            kwargs["prompt_name"] = "query"
        return model.encode(texts, **kwargs).tolist()

    # ---- Qwen3-Embedding-4B: native sentence-transformers support, last-token pooling ----
    def _embed_qwen3_4b(self, texts, is_query):
        if "qwen3_4b" not in self.loaded:
            from sentence_transformers import SentenceTransformer

            self.loaded["qwen3_4b"] = SentenceTransformer(
                "Qwen/Qwen3-Embedding-4B",
                model_kwargs={"device_map": "cuda", "torch_dtype": "bfloat16"},
            )
        model = self.loaded["qwen3_4b"]
        kwargs = {"batch_size": 16, "show_progress_bar": False}
        if is_query:
            kwargs["prompt_name"] = "query"
        return model.encode(texts, **kwargs).tolist()

    @modal.method()
    def embed(self, model_key: str, texts: list[str], is_query: bool = False):
        dispatch = {
            "specter2": self._embed_specter2,
            "qzhou": self._embed_qzhou,
            "jasper": self._embed_jasper,
            "qwen3_4b": self._embed_qwen3_4b,
        }
        if model_key not in dispatch:
            raise ValueError(f"Unknown model_key: {model_key!r} (expected one of {list(dispatch)})")

        # A single L4 (22GB usable) can't hold all 4 models at once (qzhou alone is
        # ~14GB in bf16) -- evict any other cached model before loading a new one, so
        # only the model actually in use occupies GPU memory. Repeated calls to the
        # SAME model_key (e.g. smoke test then full embed) still reuse the warm model.
        if model_key not in self.loaded:
            import gc

            import torch

            for key in list(self.loaded):
                del self.loaded[key]
            gc.collect()
            torch.cuda.empty_cache()

        return dispatch[model_key](texts, is_query)


@app.local_entrypoint()
def main(models: str = "specter2,qzhou,jasper,qwen3_4b"):
    """Standalone smoke test: `modal run scripts/modal_embeddings.py --models qzhou,jasper`."""
    worker = EmbeddingWorker()
    sample_docs = ["Carbon capture using amine sorbents. A study of CO2 removal efficiency."]
    sample_query = ["Amine-based carbon capture technologies for industrial emissions."]
    for key in models.split(","):
        doc_vec = worker.embed.remote(key, sample_docs, is_query=False)
        query_vec = worker.embed.remote(key, sample_query, is_query=True)
        print(f"{key}: doc_dim={len(doc_vec[0])} query_dim={len(query_vec[0])}")
