from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.retrieval_qa.base import RetrievalQA

def get_groq_chain_with_history(vectorstore, groq_api_key, model_name="llama3-70b-8192"):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

    prompt_template = """
    You are a helpful assistant answering questions based on documents and prior conversation.

    Previous conversation:
    {history}

    Document context:
    {context}

    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question", "history"],
        template=prompt_template,
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # This wraps the LLMChain so it can work with documents
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    # Now use it in RetrievalQA
    return RetrievalQA(retriever=vectorstore.as_retriever(), stuff_documents_chain=stuff_chain)
