# qa_chain.py
"""Builds the ConversationalRetrievalChain.

The chain supports:
  • Semantic retrieval over documents via FAISS (or any VectorStore).
  • Follow‑up‑question re‑phrasing so retrieval works even for anaphoric queries.
  • Injection of full chat history for contextual answers.

It returns a chain that expects the standard LangChain conversational input:
    {"question": <str>, "chat_history": <list[tuple[str,str]]>}
And outputs:
    {"answer": <str>}
"""

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.vectorstores.base import VectorStore

__all__ = ["get_conversational_chain"]


def get_conversational_chain(
    vectorstore: VectorStore,
    groq_api_key: str,
    model_name: str = "llama3-70b-8192",
    search_k: int = 12,
):
    """Return a fully‑wired ConversationalRetrievalChain.

    Parameters
    ----------
    vectorstore : VectorStore
        The vector store containing embedded document chunks.
    groq_api_key : str
        API key for Groq LLMs.
    model_name : str, default "llama3-70b-8192"
        Groq model to use.
    search_k : int, default 12
        Number of top similar chunks to retrieve each turn.
    """

    # ─── 1. Base LLM ──────────────────────────────────────────────────
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0,
    )

    # ─── 2. Answer‑generation prompt ─────────────────────────────────
    answer_prompt = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template="""
You are a helpful assistant answering questions about the user's document.
If the answer is not contained in the context, say “I don’t know.”

Prior conversation:
{chat_history}

Document context:
{context}

Question: {question}
Answer:
""",
    )

    answer_chain = LLMChain(llm=llm, prompt=answer_prompt)
    combine_docs_chain = StuffDocumentsChain(
        llm_chain=answer_chain,
        document_variable_name="context",
    )

    # ─── 3. Follow‑up question re‑writer ────────────────────────────
    condense_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
Given the following conversation and a follow‑up question, re‑phrase the follow‑up
question so it can be understood on its own.

Conversation so far:
{chat_history}

Follow‑up question: {question}

Standalone question:
""",
    )
    question_generator = LLMChain(llm=llm, prompt=condense_prompt)

    # ─── 4. Assemble ConversationalRetrievalChain ───────────────────
    return ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": search_k}),
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        return_source_documents=False,
    )
