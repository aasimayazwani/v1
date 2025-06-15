# ======================= vectorstore_utils.py =======================
from __future__ import annotations
import pathlib
from typing import List

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# Writable local paths
DATA_DIR   = pathlib.Path("./.data")
CACHE_PATH = DATA_DIR / "row_cache.jsonl"   # kept for potential future use
CHROMA_DIR = DATA_DIR / "chroma"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH.touch(exist_ok=True)  # harmless no-op if exists


def prepare_vectorstore(docs: List[Document], openai_key: str) -> Chroma:
    """
    Create / load a persistent Chroma vector store for the given docs.
    Embeddings are computed on the fly by OpenAIEmbeddings; no custom
    'embeddings=' kwarg is passed.
    """
    embedder = OpenAIEmbeddings(openai_api_key=openai_key)

    return Chroma.from_documents(
        documents=docs,
        embedding=embedder,                # <— correct kwarg
        persist_directory=str(CHROMA_DIR),
    )
