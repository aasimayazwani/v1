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
    """Return a ConversationalRetrievalChain that accepts
    {'question': str, 'chat_history': List[Tuple[str, str]]}
    and outputs {'answer': str}.
    """
    # 1. Groq LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0,
    )

    # 2. Prompt that expects chat_history + question
    prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
You are a helpful assistant answering questions about the user's document.
If you don’t know the answer, say “I don’t know” instead of inventing one.

Prior chat history:
{chat_history}

Document context:
{context}

Question: {question}
Answer:
""",
    )

    # 3. Chain that stuffs retrieved docs into {context}
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
    )

    # 4. ConversationalRetrievalChain wires everything together
    return ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        combine_docs_chain=stuff_chain,
        return_source_documents=False,   # flip to True if you want sources
    )
