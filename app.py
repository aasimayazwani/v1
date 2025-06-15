# ======================= app.py =======================
import streamlit as st
from uuid import uuid4
from typing import List, Tuple

from file_handlers import parse_pdf, parse_csv
from vectorstore_utils import prepare_vectorstore
from df_agent_utils import build_sql_agent, build_wikipedia_agent
from qa_chain import build_rag_chain
from db_utils import init_db, save_chat, load_chat

# ─── Page config & secrets ───────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("📄🔍 CSV & PDF Chatbot with SQL + Wikipedia")

openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
groq_api_key   = st.secrets.get("GROQ_API_KEY", "")

if not (openai_api_key and groq_api_key):
    st.error("Add OPENAI_API_KEY and GROQ_API_KEY to `.streamlit/secrets.toml`.")
    st.stop()

# ─── Persistent session & DB ─────────────────────────────────────────
init_db()
session_id = st.session_state.get("session_id") or str(uuid4())
st.session_state["session_id"] = session_id
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat(session_id)  # [(role,text)]

# ─── Sidebar (upload & settings) ─────────────────────────────────────
with st.sidebar:
    st.header("Upload document")
    uploaded_file = st.file_uploader("CSV or PDF", type=["csv", "pdf"])
    k_chunks      = st.slider("Retriever top-k", 4, 20, 12)
    show_src      = st.checkbox("Show source docs (RAG)", value=False)

# ─── Always-on Wikipedia agent ───────────────────────────────────────
wiki_agent = build_wikipedia_agent(groq_api_key)

# ─── Build doc-dependent resources ───────────────────────────────────
df_agent   = None   # SQL agent (DuckDB)
rag_chain  = None   # FAISS RAG chain

if uploaded_file:
    if uploaded_file.name.lower().endswith(".pdf"):
        docs = [parse_pdf(uploaded_file)]
    else:
        df, docs = parse_csv(uploaded_file)        # df + list[Document]
        df_agent = build_sql_agent(df, groq_api_key)

    with st.spinner("Embedding & indexing …"):
        vs        = prepare_vectorstore(docs, openai_api_key)
        rag_chain = build_rag_chain(vs, groq_api_key, k_chunks)
    st.sidebar.success("Document indexed ✅")
else:
    st.sidebar.info("Chat with Wikipedia until you upload a file.")

# ─── Replay previous conversation bubbles ────────────────────────────
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)

# ─── Chat loop ───────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question …"):
    st.chat_message("user").markdown(prompt)

    # RAG memory (tuples)
    tuples: List[Tuple[str, str]] = [
        (st.session_state.chat_history[i][1], st.session_state.chat_history[i + 1][1])
        for i in range(0, len(st.session_state.chat_history), 2)
        if i + 1 < len(st.session_state.chat_history)
    ]

    answer = None

    # 1️⃣ Wikipedia prefix
    if prompt.lower().startswith("wiki:"):
        answer = wiki_agent.run(prompt[5:].strip())

    # 2️⃣ Try SQL agent for any CSV
    if answer is None and df_agent:
        try:
            answer = df_agent.run(prompt)
        except Exception:
            answer = None

    # 3️⃣ RAG over document (PDF or CSV text)
    if answer is None and rag_chain:
        answer = rag_chain.invoke({"input": prompt, "chat_history": tuples})["answer"]

    # 4️⃣ Fallback
    if answer is None:
        answer = "Sorry, I couldn't answer that."

    assistant = st.chat_message("assistant")
    assistant.markdown(answer)

    # Show source docs (optional)
    if show_src and isinstance(answer, dict) and "source_documents" in answer:
        with assistant.expander("Sources"):
            st.write(answer["source_documents"])

    # Persist
    st.session_state.chat_history += [("user", prompt), ("assistant", answer)]
    save_chat(session_id, prompt, answer)
