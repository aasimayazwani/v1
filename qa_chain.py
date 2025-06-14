from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.retrieval_qa.base import RetrievalQA

def get_groq_chain_with_history(vectorstore,
                                groq_api_key,
                                model_name="llama3-70b-8192"):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

    prompt = PromptTemplate(
        template="""
You are a helpful assistant answering questions about the uploaded document.

Previous conversation:
{history}

Context:
{context}

Question: {question}
Answer:
""",
        input_variables=["history", "question"],
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,           # runs the prompt
        document_variable_name="context"  # where StuffDocumentsChain injects docs
    )

    return RetrievalQA(
        retriever=vectorstore.as_retriever(),
        combine_documents_chain=combine_documents_chain,
    )
