# ======================= file_handlers.py =======================
import pandas as pd
from langchain.docstore.document import Document

MAX_ROWS = 50_000  # guard cost


def parse_csv(file):
    df = pd.read_csv(file, nrows=MAX_ROWS)
    docs = [
        Document(
            page_content=" | ".join(f"{col}: {val}" for col, val in row.items()),
            metadata={"row_index": int(i)},
        )
        for i, row in df.iterrows()
    ]
    return df, docs


def parse_pdf(file):
    from PyPDF2 import PdfReader
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# ======================= vectorstore_utils.py =======================
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import hashlib, json, os, pathlib

CACHE_PATH = pathlib.Path("/mnt/data/embed_cache.jsonl")
CHROMA_PATH = "/mnt/data/chroma"


def _cached_vector(text: str, embedder) -> list[float]:
    h = hashlib.sha256(text.encode()).hexdigest()
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            for line in f:
                rec = json.loads(line)
                if rec["h"] == h:
                    return rec["v"]
    v = embedder.embed_query(text)
    with open(CACHE_PATH, "a") as f:
        f.write(json.dumps({"h": h, "v": v}) + "\n")
    return v


def prepare_vectorstore(docs: list[Document], openai_key: str):
    embedder = OpenAIEmbeddings(openai_api_key=openai_key)
    vectors = [_cached_vector(d.page_content, embedder) for d in docs]
    return Chroma.from_documents(docs, vectors, persist_directory=CHROMA_PATH)