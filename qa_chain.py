# ======================= qa_chain.py =======================
"""
Minimal RAG chain that works on virtually any LangChain build >= 0.1.17.
No history-aware helper, no exotic imports—just:
  • LLM (Groq)
  • VectorStore retriever
  • Stuff-documents answer chain
Exposed:
    build_rag_chain(vectorstore, groq_api_key, top_k=12) -> chain
Chain expects:
    {"input": question, "chat_history": [(q,a), ...]}  # chat_history ignored
Returns:
    {"answer": str, "source_documents": [Document, ...]}
"""
from langchain_groq import ChatGroq
from langchain.vectorstores.base import VectorStore
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant answering questions about documents.
Use ONLY the context below. If unsure, say "I don't know."

Context:
{context}

Question: {input}
Helpful answer:"""
)


def build_rag_chain(
    vectorstore: VectorStore,
    groq_api_key: str,
    top_k: int = 12,
    model_name: str = "llama3-70b-8192",
):
    # Base LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0)

    # Simple retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )

    # Answer-generation chain
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=ANSWER_PROMPT)

    # Final RAG pipeline
    return create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
    )
