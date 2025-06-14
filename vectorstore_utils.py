# vectorstore_utils.py
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def prepare_vectorstore(docs, openai_api_key):
    """
    docs : list[langchain.docstore.document.Document]  (one per CSV row or PDF chunk)
    """
    embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_documents(docs, embedder)
