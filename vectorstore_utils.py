from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def prepare_vectorstore(text, openai_api_key):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_texts(chunks, embedder)
