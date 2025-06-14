"""qa_chain.py – Modern RAG builder (LangChain ≥ 0.2)

Exposes:
    build_rag_chain(vectorstore, groq_key, k=12) → Runnable

The returned chain expects a dict:
    {"input": <question>, "chat_history": <[(q,a), …]>}
And returns:
    {"answer": <str>, "source_documents": [Document, …]}
"""
from langchain_groq import ChatGroq
from langchain.retrievers import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_core.vectorstores import VectorStore


def build_rag_chain(vectorstore: VectorStore, groq_key: str, k: int = 12):
    """Create a retrieval‑augmented generation chain using Groq LLM.

    Parameters
    ----------
    vectorstore : VectorStore
        A langchain vector store with embedded docs.
    groq_key : str
        Groq API key.
    k : int
        Number of similar chunks to retrieve per query.
    """
    llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=groq_key, temperature=0)

    retriever = create_history_aware_retriever(
        llm,
        vectorstore.as_retriever(search_kwargs={"k": k}),
    )

    rag_chain = create_retrieval_chain(retriever, llm, return_source_documents=True)
    return rag_chain
