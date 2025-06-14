# ======================= app.py =======================
import streamlit as st
from uuid import uuid4
from typing import List, Tuple
from file_handlers import parse_pdf, parse_csv
from vectorstore_utils import prepare_vectorstore
from df_agent_utils import build_sql_agent
from qa_chain import build_rag_chain
from db_utils import init_db, save_chat, load_chat

st.set_page_config(layout="wide")
st.title("📄🔍 Groq‑Powered CSV & PDF Chatbot")

# --- secrets & rate‑limit ---------------------------------------------------
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
groq_api_key   = st.secrets.get("GROQ_API_KEY",   "")

MAX_UPLOAD_MB  = 10

# --- persistent session -----------------------------------------------------
init_db()
user_id    = st.session_state.get("session_id") or str(uuid4())
st.session_state["session_id"] = user_id
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat(user_id)  # [(role, text)]

# --- sidebar: upload + settings --------------------------------------------
with st.sidebar:
    st.header("🗂️ Document")
    uploaded_file = st.file_uploader("CSV or PDF", type=["csv", "pdf"])
    st.markdown("---")
    k_chunks = st.slider("Retriever top‑k", 3, 20, 12)
    show_sources = st.checkbox("Show sources", value=True)

if not (uploaded_file and openai_api_key and groq_api_key):
    st.info("Upload a file and set API keys in secrets to begin.")
    st.stop()

# --- Parse & embed ----------------------------------------------------------
if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"File exceeds {MAX_UPLOAD_MB} MB limit.")
    st.stop()

if uploaded_file.name.lower().endswith(".pdf"):
    raw_text = parse_pdf(uploaded_file)
    docs, df, sql_agent = [raw_text], None, None
else:
    df, docs = parse_csv(uploaded_file)
    sql_agent = build_sql_agent(df, groq_api_key) if df is not None else None

with st.spinner("🔎 Embedding & indexing …"):
    vs = prepare_vectorstore(docs, openai_api_key)
    rag_chain = build_rag_chain(vs, groq_api_key, k_chunks)

# --- replay bubbles ---------------------------------------------------------
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# --- chat loop --------------------------------------------------------------
if prompt := st.chat_input("Ask about the document …"):
    st.chat_message("user").markdown(prompt)

    # Convert bubble list → (q,a) tuples for RAG
    tuples: List[Tuple[str, str]] = [
        (st.session_state.chat_history[i][1], st.session_state.chat_history[i+1][1])
        for i in range(0, len(st.session_state.chat_history), 2)
        if i+1 < len(st.session_state.chat_history)
    ]

    # naive analytics keyword detect
    analytic = any(word in prompt.lower() for word in (
        "average", "mean", "sum", "count", "plot", "histogram", "max", "min"))

    answer = None
    if analytic and sql_agent is not None:
        try:
            answer = sql_agent.run(prompt)
        except Exception:
            answer = None

    if answer is None:
        answer = rag_chain.invoke({
            "input": prompt,
            "chat_history": tuples,
        })["answer"]
        # fallback: let SQL answer unknowns
        if ("i don't know" in answer.lower()) and sql_agent is not None:
            try:
                answer = sql_agent.run(prompt)
            except Exception:
                pass

    assistant_container = st.chat_message("assistant")
    assistant_container.markdown(answer)

    # optional source documents
    if show_sources and "source_documents" in answer:
        with assistant_container.expander("Sources"):
            st.write(answer["source_documents"])

    st.session_state.chat_history += [("user", prompt), ("assistant", answer)]
    save_chat(user_id, prompt, answer)