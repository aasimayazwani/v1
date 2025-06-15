# ======================= qa_chain.py =======================
from langchain_groq import ChatGroq

from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStore
from langchain.output_parsers import StrOutputParser
from langchain.schema.document import Document

from langchain.retrievers import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


def build_rag_chain(
    vectorstore: VectorStore,
    groq_api_key: str,
    top_k: int = 12,
    model: str = "llama3-70b-8192",
):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    # --- History-aware retriever ---
    contextual_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=PromptTemplate.from_template(
            """You are an assistant helping users with questions about their documents.
Given the conversation history and a follow-up question, rewrite the question to be standalone.

Chat history:
{chat_history}

Follow-up question: {input}
Standalone question:"""
        ),
    )

    # --- Answer generation ---
    answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=PromptTemplate.from_template(
            """You are a helpful assistant for answering questions about documents.
Use only the following pieces of context to answer the question. If unsure, say "I don't know."

Context:
{context}

Question: {input}
"""
        ),
    )

    return create_retrieval_chain(
        retriever=contextual_retriever,
        combine_docs_chain=answer_chain,
    )
