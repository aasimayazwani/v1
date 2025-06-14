from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

def get_groq_chain(vectorstore, groq_api_key, model_name="mixtral-8x7b-32768"):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")