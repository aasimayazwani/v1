# ======================= qa_chain.py =======================
"""
RAG builder that works whether or not
`create_history_aware_retriever` is present.

Usage:
    chain = build_rag_chain(vectorstore, groq_api_key, top_k=12)
    result = chain.invoke({"input": question, "chat_history": [(q,a), ...]})
"""

from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain.vectorstores.base import VectorStore
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser

# Try to import the official helper (exists in some 0.2 builds)
try:
    from langchain.retrievers.history_aware import create_history_aware_retriever

    _HAVE_OFFICIAL = True
except ImportError:  # fall back to custom implementation
    from langchain.retrievers.vectorstore import VectorStoreRetriever
    from langchain_core.runnables import RunnableMap, RunnablePassthrough

    _HAVE_OFFICIAL = False


# ── Prompt templates --------------------------------------------------------
CONDENSE_PROMPT = PromptTemplate.from_template(
    """You are an assistant helping users with questions about their documents.
Given the conversation history and a follow-up question, rewrite the question
so it can be understood on its own.

Chat history:
{chat_history}

Follow-up question: {input}
Standalone question:"""
)

ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant for answering questions about documents.
Use ONLY the following pieces of context to answer. If unsure, say "I don't know."

Context:
{context}

Question: {input}
Helpful answer:"""
)


# ── Builder -----------------------------------------------------------------
def build_rag_chain(
    vectorstore: VectorStore,
    groq_api_key: str,
    top_k: int = 12,
    model_name: str = "llama3-70b-8192",
):
    """Return a RAG chain that is history-aware when possible."""
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    if _HAVE_OFFICIAL:
        # Use the official helper
        history_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=CONDENSE_PROMPT,
        )
    else:
        # --- Custom minimal history-aware retriever ------------------------
        # 1) chain to rewrite the question
        rewrite_chain = (
            {
                "chat_history": lambda x: x["chat_history"],
                "input": lambda x: x["input"],
            }
            | CONDENSE_PROMPT
            | llm
            | StrOutputParser()
        )

        # 2) wrap retriever so it accepts the rewritten question
        def _retrieve(inputs: Dict[str, Any]):
            new_q = rewrite_chain.invoke(inputs)
            return retriever.get_relevant_documents(new_q)

        class _HistoryAwareRetriever(VectorStoreRetriever):
            def get_relevant_documents(self, query: str):
                # query contains both 'input' & 'chat_history' via RunnableMap
                return _retrieve(query)

        history_retriever = RunnableMap(
            {"context": _retrieve, "input": lambda x: x["input"]}
        )

    # Chain that stuffs docs + answers
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=ANSWER_PROMPT,
    )

    return create_retrieval_chain(
        retriever=history_retriever,
        combine_docs_chain=combine_docs_chain,
    )
