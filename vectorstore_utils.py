from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import hashlib, json, os

EMBED_CACHE = "/mnt/data/row_cache.jsonl"

def _cached_embed(text, embedder):
    h = hashlib.sha256(text.encode()).hexdigest()
    if os.path.exists(EMBED_CACHE):
        with open(EMBED_CACHE) as f:
            for line in f:
                row = json.loads(line);     # {"h":..,"v":[..]}
                if row["h"] == h: return row["v"]
    v = embedder.embed_query(text)
    with open(EMBED_CACHE, "a") as f:
        f.write(json.dumps({"h": h, "v": v}) + "\n")
    return v

def prepare_vectorstore(documents, openai_key):
    embedder = OpenAIEmbeddings(openai_api_key=openai_key)
    texts  = [d.page_content for d in documents]
    vectors = [_cached_embed(t, embedder) for t in texts]
    return Chroma.from_documents(documents, embeddings=vectors, persist_directory="/mnt/data/chroma")
