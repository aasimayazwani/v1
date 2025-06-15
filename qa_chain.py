"""Conversational RAG chain (works on LangChain 0.1.x)."""
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.retrieval_qa.base import ConversationalRetrievalChain
from langchain.vectorstores.base import VectorStore


def build_rag_chain(vs: VectorStore, groq_key: str, k=12, model="llama3-70b-8192"):
    llm = ChatGroq(groq_api_key=groq_key, model_name=model, temperature=0)

    answer_prompt = PromptTemplate(
        template=(
            "You answer based on the context. If unsure say \"I don't know.\""
            "\n\nChat history:\n{chat_history}"
            "\n\nContext:\n{context}"
            "\n\nQuestion: {question}\nAnswer:"
        ),
        input_variables=["chat_history", "question", "context"],
    )
    answer_chain = LLMChain(llm=llm, prompt=answer_prompt)
    combine_chain = StuffDocumentsChain(
        llm_chain=answer_chain, document_variable_name="context"
    )

    condense_prompt = PromptTemplate(
        template=(
            "Rewrite the follow-up question to be standalone.\n"
            "Chat so far:\n{chat_history}\n\nFollow-up: {question}\n\nStandalone:"
        ),
        input_variables=["chat_history", "question"],
    )
    q_gen = LLMChain(llm=llm, prompt=condense_prompt)

    return ConversationalRetrievalChain(
        retriever=vs.as_retriever(search_kwargs={"k": k}),
        combine_docs_chain=combine_chain,
        question_generator=q_gen,
        return_source_documents=False,
    )
