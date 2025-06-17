"""
vectorstore_utils.py
────────────────────────────────────────────────────────────────────────────
Utility wrapper around one (or two) LangChain-compatible vector stores so
the rest of your code can call a single API regardless of the backend.

✓  Keeps existing behaviour: default FAISS, optional Chroma.
✓  Adds:
      • add_pdf_docs(list[Document])            – quick bulk insert for PDFs
      • add_gtfs_dict(dict[str, DataFrame])     – store GTFS tables in
                                                  separate namespaces
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma

# ───────────────────────────────────────────────────────────────────────────
# Configuration constants (tweak as you like)
# ───────────────────────────────────────────────────────────────────────────
_DEFAULT_DIM = 1536
_FAISS_PERSIST = Path(".vectordb/faiss")
_CHROMA_PERSIST = Path(".vectordb/chroma")


# ───────────────────────────────────────────────────────────────────────────
# Main wrapper
# ───────────────────────────────────────────────────────────────────────────
class VectorStore:
    """
    A tiny façade over FAISS or Chroma so caller code doesn’t need to care.

    >>> vs = VectorStore("faiss")
    >>> vs.add_documents(docs)
    >>> results = vs.similarity_search("where is my bus?", k=4)
    """

    def __init__(self, provider: str = "faiss", **provider_kwargs):
        self.provider = provider.lower()
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        if self.provider == "faiss":
            self._store = FAISS(
                embedding_function=self._embeddings,
                index=None,
                **provider_kwargs,
            )
        elif self.provider == "chroma":
            self._store = Chroma(
                collection_name="default",
                embedding_function=self._embeddings,
                persist_directory=str(_CHROMA_PERSIST),
                **provider_kwargs,
            )
        else:
            raise ValueError("provider must be 'faiss' or 'chroma'")

    # ──────────────────────────────────
    # Passthroughs to underlying store
    # ──────────────────────────────────
    def add_documents(self, docs: List[Document], namespace: str | None = None):
        """Generic bulk insert (kept for backward compat)."""
        self._store.add_documents(docs, namespace=namespace)

    def similarity_search(self, query: str, k: int = 4):
        return self._store.similarity_search(query, k=k)

    def as_retriever(self, **kwargs):
        return self._store.as_retriever(**kwargs)

    def persist(self):
        """Persist if backend supports it (e.g. Chroma)."""
        if hasattr(self._store, "persist"):
            self._store.persist()

    # ──────────────────────────────────
    # NEW convenience helpers
    # ──────────────────────────────────
    def add_pdf_docs(self, docs: List[Document]):
        """
        Shortcut so callers can simply do:

            vs.add_pdf_docs(pdf_docs)
        """
        self.add_documents(docs)

    def add_gtfs_dict(self, gtfs: Dict[str, pd.DataFrame]):
        """
        Store each GTFS table in its own namespace (e.g., 'gtfs::stops.txt') so
        they’re easy to filter during retrieval.

        Parameters
        ----------
        gtfs : dict[str, pd.DataFrame]
            Output of _load_gtfs_zip from `file_handlers.py`.
        """
        for table_name, df in gtfs.items():
            # Convert every row to its own embedding-friendly Document
            row_docs = [
                Document(
                    page_content=row.to_json(),
                    metadata={"table": table_name, "row": idx},
                )
                for idx, row in df.iterrows()
            ]
            self.add_documents(row_docs, namespace=f"gtfs::{table_name}")

    # ──────────────────────────────────
    # Optional: expose underlying store
    # ──────────────────────────────────
    @property
    def backend(self):
        return self._store
