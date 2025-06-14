# ======================= qa_chain.py =======================
"""RAG chain using new LangChain v0.2 helpers."""
from langchain_groq import ChatGroq
from langchain.retrievers import create_history_aware_retriever
from langchain.chains import create_retrieval_chain


def build_rag_chain(vectorstore, groq_key, k: int):
    llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=groq_key, temperature=0)
    retriever = create_history_aware_retriever(llm, vectorstore.as_retriever(search_kwargs={"k": k}))
    return create_retrieval_chain(retriever, llm)