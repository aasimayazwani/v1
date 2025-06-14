# ======================= qa_chain.py =======================
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import RunnableMap

from langchain_core.retrievers import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Prompt Templates -------------------------------------------------------

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {input}
Standalone question:""")

ANSWER_PROMPT = PromptTemplate.from_template("""
You are a helpful AI assistant. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.

Context:
{context}

Question: {input}
Helpful Answer:""")

# --- Chain builder ----------------------------------------------------------

def build_rag_chain(
    vectorstore: VectorStore,
    groq_api_key: str,
    top_k: int = 10,
    model_name: str = "llama3-70b-8192"
) -> RunnableMap:
    """
    Builds a retrieval-augmented generation (RAG) chain.
    Returns a LangChain Runnable that accepts a dict:
        { "input": question, "chat_history": [(q,a), (q,a), …] }
    Returns a dict with:
        { "answer": <str>, "source_documents": [Document, …] }
    """

    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=CONDENSE_QUESTION_PROMPT,
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=ANSWER_PROMPT
    )

    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_docs_chain
    )

    return rag_chain
