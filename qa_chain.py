"""
qa_chain.py
────────────────────────────────────────────────────────────────────────────
Builds a Retrieval-Augmented-Generation (RAG) chain with LangChain.

▸ Original behaviour (kept)
    • build_qa_chain(vstore, llm) – returns RetrievalQA chain.

▸ New convenience
    • answer(query, source, llm)  – one-shot helper that accepts either:
        – list[Document]  (raw PDF chunks)   → auto-indexes to FAISS
        – VectorStore     (already indexed)  → uses as-is
      and returns (answer_text, source_documents).
"""

from __future__ import annotations

from typing import List, Tuple, Union

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.schema import Document, BaseLanguageModel

from vectorstore_utils import VectorStore


# ───────────────────────────────────────────────────────────────────────────
# 1.  Original helper (unchanged)
# ───────────────────────────────────────────────────────────────────────────
def build_qa_chain(
    vstore: VectorStore,
    llm: BaseLanguageModel | None = None,
) -> RetrievalQA:
    """
    Construct a LangChain RetrievalQA chain that pulls k=4 relevant
    docs from `vstore` and feeds them to the LLM in a Stuff chain.
    """
    if llm is None:
        llm = OpenAI(temperature=0, max_tokens=512)

    retriever = vstore.as_retriever(search_kwargs={"k": 4})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return chain


# ───────────────────────────────────────────────────────────────────────────
# 2.  NEW – high-level helper
# ───────────────────────────────────────────────────────────────────────────
def answer(
    query: str,
    source: Union[List[Document], VectorStore],
    llm: BaseLanguageModel | None = None,
) -> Tuple[str, List[Document]]:
    """
    Convenience wrapper so UI code can do:

        text, sources = answer(user_q, pdf_docs, llm)
        #  – OR –
        text, sources = answer(user_q, existing_vstore, llm)

    Parameters
    ----------
    query  : str
        The user’s natural-language question.
    source : list[Document] | VectorStore
        Either raw chunks (e.g. from _load_pdf) or an indexed store.
    llm    : BaseLanguageModel | None
        Optionally pass the same LLM you use elsewhere; otherwise an
        OpenAI `gpt-3.5-turbo-0125` with temperature 0 is created.

    Returns
    -------
    tuple[str, list[Document]]
        The answer text and the documents LangChain selected.
    """
    # ------------------------------------------------------------------ #
    # 1) Ensure we have a VectorStore to search
    # ------------------------------------------------------------------ #
    if isinstance(source, list):               # raw Document chunks
        vstore = VectorStore("faiss")          # in-mem FAISS index
        vstore.add_pdf_docs(source)
    elif isinstance(source, VectorStore):      # already a vector DB
        vstore = source
    else:
        raise TypeError(
            "`source` must be either list[Document] or VectorStore, "
            f"not {type(source)}"
        )

    # ------------------------------------------------------------------ #
    # 2) Build / run the RAG chain
    # ------------------------------------------------------------------ #
    qa = build_qa_chain(vstore, llm)
    result = qa.invoke({"query": query})

    return result["result"], result["source_documents"]
