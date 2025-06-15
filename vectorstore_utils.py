# ======================= vectorstore_utils.py =======================
from __future__ import annotations
import hashlib, json, pathlib
from typing import List

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Writable cache directory
DATA_DIR   = pathlib.Path("./.data")
CACHE_PATH = DATA_DIR / "row_cache.jsonl"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH.touch(exist_ok=True)


def _cached_embed(text: str, embedder: OpenAIEmbeddings) -> List[float]:
    """Return embedding for `text`, using SHA-256 cache to avoid recompute."""
    h = hashlib.sha256(text.encode()).hexdigest()
    with CACHE_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["h"] == h:
                return rec["v"]

    vec = embedder.embed_query(text)
    with CACHE_PATH.open("a") as f:
        f.write(json.dumps({"h": h, "v": vec}) + "\n")
    return vec


def prepare_vectorstore(docs: List[Document], openai_key: str) -> FAISS:
    """
    Build an in-memory FAISS vector store from `docs`.
    Uses OpenAI embeddings with local caching.
    """
    embedder = OpenAIEmbeddings(openai_api_key=openai_key)

    # Trigger embedding once so cache is populated (optional)
    for d in docs:
        _cached_embed(d.page_content, embedder)

    # Let FAISS compute (it will hit the cache, so duplicate work is avoided)
    return FAISS.from_documents(docs, embedder)
