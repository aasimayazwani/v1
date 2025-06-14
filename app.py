# ======================= app.py =======================
import streamlit as st
from uuid import uuid4
from typing import List, Tuple

from file_handlers import parse_pdf, parse_csv
from vectorstore_utils import prepare_vectorstore
from df_agent_utils import build_sql_agent, build_wikipedia_agent
from qa_chain import build_rag_chain
from db_utils import init_db, save_chat, load_chat

# ─── Config & secrets ───────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("📄🔍 Multi-Source Document Chatbot")
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
groq_api_key   = st.secrets.get("GROQ_API_KEY", "")

if not (openai_api_key and groq_api_key):
    st.warning("Add OPENAI_API_KEY and GROQ_API_KEY in secrets!")
    st.stop()

# ─── Session & DB ───────────────────────────────────────────────────────────
init_db()
user_id = st.session_state.get("session_id") or str(uuid4())
st.session_state["session_id"] = user_id
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat(user_id)  # [(role, text)]

# ─── Sidebar UI ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload document")
    uploaded_file = st.file_uploader("CSV or PDF", type=["csv", "pdf"])
    k_chunks = st.slider("Retriever k", 3, 20, 12)
    show_src = st.checkbox("Show sources", True)

# ─── Build always-available Wikipedia agent ────────────────────────────────
wiki_agent = build_wikipedia_agent(groq_api_key)

# ─── Document-dependent resources ──────────────────────────────────────────
df_agent = None
rag_chain = None
if uploaded_file:
    if uploaded_file.name.lower().endswith(".pdf"):
        docs = [parse_pdf(uploaded_file)]
    else:
        df, docs = parse_csv(uploaded_file)
        df_agent = build_sql_agent(df, groq_api_key)
    vs = prepare_vectorstore(docs, openai_api_key)
    rag_chain = build_rag_chain(vs, groq_api_key, k_chunks)
    st.sidebar.success("Document indexed ✅")
else:
    st.sidebar.info("Chat with Wikipedia until you upload a file.")

# ─── Replay previous conversation bubbles ─────────────────────────────────
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)

# ─── Chat loop ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question …"):
    st.chat_message("user").markdown(prompt)

    # History → (question, answer) tuples for memory-aware RAG
    tuples: List[Tuple[str, str]] = [
        (st.session_state.chat_history[i][1], st.session_state.chat_history[i + 1][1])
        for i in range(0, len(st.session_state.chat_history), 2)
        if i + 1 < len(st.session_state.chat_history)
    ]

    answer = None
    # 1️⃣ Explicit Wikipedia prefix
    if prompt.lower().startswith("wiki:"):
        answer = wiki_agent.run(prompt[len("wiki:"):].strip())

    # 2️⃣ Analytics keywords → DuckDB (if CSV present)
    analytic_kw = ("average", "mean", "sum", "count", "plot", "hist", "min", "max")
    if answer is None and df_agent and any(k in prompt.lower() for k in analytic_kw):
        try:
            answer = df_agent.run(prompt)
        except Exception:
            answer = None

    # 3️⃣ RAG over uploaded document
    if answer is None and rag_chain:
        answer = rag_chain.invoke({"input": prompt, "chat_history": tuples})["answer"]

    # 4️⃣ Final fallback → Wikipedia
    if answer is None:
        answer = wiki_agent.run(prompt)

    assistant = st.chat_message("assistant")
    assistant.markdown(answer)

    if show_src and rag_chain and "source_documents" in answer:
        with assistant.expander("Sources"):
            st.write(answer["source_documents"])

    st.session_state.chat_history += [("user", prompt), ("assistant", answer)]
    save_chat(user_id, prompt, answer)
