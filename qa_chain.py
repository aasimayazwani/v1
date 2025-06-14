# qa_chain.py
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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

    chain = LLMChain(llm=llm, prompt=prompt)
    return RetrievalQA(combine_documents_chain=chain, retriever=vectorstore.as_retriever())