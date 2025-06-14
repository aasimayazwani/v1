# qa_chain.py
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


def get_conversational_chain(
    vectorstore,
    groq_api_key: str,
    model_name: str = "llama3-70b-8192",
):
    """
    Build a ConversationalRetrievalChain that accepts:
        {"question": <str>, "chat_history": <List[Tuple[str, str]]>}
    and returns:
        {"answer": <str>}
    """

    # ────────────────────────────────
    # 1)  LLM (Groq) for all sub-chains
    # ────────────────────────────────
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0,
    )

    # ────────────────────────────────
    # 2)  Prompt for generating answers
    # ────────────────────────────────
    answer_prompt = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template="""
You are a helpful assistant answering questions about the user's document.
If the answer is not in the document, say “I don’t know.”

Previous conversation:
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

    # ────────────────────────────────
    # 3)  Prompt for re-phrasing follow-up questions
    # ────────────────────────────────
    condense_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
Given the following conversation and a follow-up question,
re-phrase the follow-up question so that it can be understood
by itself (i.e. a standalone question).

Conversation so far:
{chat_history}

Follow-up question: {question}

Standalone question:
""",
    )
    question_generator = LLMChain(llm=llm, prompt=condense_prompt)

    # ────────────────────────────────
    # 4)  Assemble ConversationalRetrievalChain
    # ────────────────────────────────
    return ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        return_source_documents=False,
    )
