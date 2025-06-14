from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

def get_groq_chain(vectorstore, groq_api_key, model_name="llama3-70b-8192"):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

    prompt_template = """
    Use the following context to answer the question.
    If you don’t know the answer, just say “I don’t know.” Don’t try to make up an answer.

    Context:
    {context}

    Question: {question}
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return RetrievalQA(combine_documents_chain=qa_chain, retriever=vectorstore.as_retriever())
