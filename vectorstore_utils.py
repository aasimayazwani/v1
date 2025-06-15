# ======================= vectorstore_utils.py =======================
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import hashlib, json, os, pathlib

CACHE_PATH = pathlib.Path("/mnt/data/row_cache.jsonl")
CHROMA_PATH = "/mnt/data/chroma"

# ── Ensure paths exist -------------------------------------------------
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)      # ← add this
if not CACHE_PATH.exists():                              # ← add this
    CACHE_PATH.touch()

def _cached_embed(text: str, embedder) -> list[float]:
    h = hashlib.sha256(text.encode()).hexdigest()
    with CACHE_PATH.open() as f:                          # streamlined check
        for line in f:
            rec = json.loads(line)
            if rec["h"] == h:
                return rec["v"]

    v = embedder.embed_query(text)
    with CACHE_PATH.open("a") as f:
        f.write(json.dumps({"h": h, "v": v}) + "\n")
    return v

def prepare_vectorstore(docs: list[Document], openai_key: str):
    embedder = OpenAIEmbeddings(openai_api_key=openai_key)
    vectors = [_cached_embed(d.page_content, embedder) for d in docs]
    return Chroma.from_documents(docs, vectors, persist_directory=CHROMA_PATH)
