# ======================= vectorstore_utils.py =======================
from __future__ import annotations
import hashlib, json, pathlib
from typing import List

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# Writable local paths
DATA_DIR   = pathlib.Path("./.data")
CACHE_PATH = DATA_DIR / "row_cache.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH.touch(exist_ok=True)

def _cached_embed(text: str, embedder: OpenAIEmbeddings) -> List[float]:
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

def prepare_vectorstore(docs: List[Document], openai_key: str) -> Chroma:
    embedder = OpenAIEmbeddings(openai_api_key=openai_key)
    vectors  = [_cached_embed(d.page_content, embedder) for d in docs]
    return Chroma.from_documents(
        documents=docs,
        embeddings=vectors,
        persist_directory=str(CHROMA_DIR),
    )
