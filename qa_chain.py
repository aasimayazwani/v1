# ========================== qa_chain.py ==========================
"""
Builds a Retrieval-Augmented Generation (RAG) chain compatible with
LangChain 0.2.x.

Exposed function:
    build_rag_chain(vectorstore, groq_api_key, top_k=12) -> chain

The returned chain expects:
    {"input": <question>, "chat_history": <[(q,a), …]>}
and returns:
    {"answer": <str>, "source_documents": [Document, ...]}  (the key appears
     only if you enable it when composing the chain).
"""
from langchain_groq import ChatGroq
from langchain.vectorstores.base import VectorStore
from langchain.prompts import PromptTemplate

# New helper builders (0.2.x)
from langchain.retrievers.history_aware import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


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
Use ONLY the following pieces of context to answer. If you are unsure,
say "I don't know."

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
    """
    Return a Runnable chain that performs:
      1. Question re-write (history aware)
      2. Vector retrieval (similarity top-k)
      3. Stuff-documents answer generation
    """

    # Base LLM (Groq)
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0,
    )

    # Retriever with memory-aware rewrite
    basic_retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k, "search_type": "similarity"}
    )

    contextual_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=basic_retriever,
        prompt=CONDENSE_PROMPT,
    )

    # Chain that stuffs docs + answers
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=ANSWER_PROMPT,
    )

    # Final RAG pipeline
    rag_chain = create_retrieval_chain(
        retriever=contextual_retriever,
        combine_docs_chain=combine_docs_chain,
    )

    return rag_chain
