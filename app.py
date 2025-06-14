# app.py
import streamlit as st
import uuid

from file_handlers import parse_pdf, parse_csv
from vectorstore_utils import prepare_vectorstore
from qa_chain import get_conversational_chain
from db_utils import init_db, save_chat, load_chat

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("📄🔍 Groq-Powered Document Q&A Bot")

openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
groq_api_key   = st.secrets.get("GROQ_API_KEY",   "")

# ─── Persistent session id + DB ──────────────────────────────────────────────
init_db()
session_id = st.session_state.get("session_id") or str(uuid.uuid4())
st.session_state["session_id"] = session_id
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat(session_id)  # List[(q, a)]

# ─── File upload ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a CSV or PDF", type=["csv", "pdf"])

if uploaded_file and openai_api_key and groq_api_key:
    # 1. Extract raw text
    text = parse_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else parse_csv(uploaded_file)

    # 2. Vector store + conversational chain
    with st.spinner("Embedding & indexing…"):
        vs        = prepare_vectorstore(text, openai_api_key)
        qa_chain  = get_conversational_chain(vs, groq_api_key)
    st.success("Chatbot is ready!")

    # ─── Chat UI ─────────────────────────────────────────────────────────────
    query = st.text_input("Ask something about your document:")
    if query:
        # 3. Call chain with proper keys
        result  = qa_chain.invoke({
            "question":     query.strip(),
            "chat_history": st.session_state.chat_history,   # List[(q, a)]
        })
        answer  = result["answer"]

        # 4. Persist + show
        st.session_state.chat_history.append((query, answer))
        save_chat(session_id, query, answer)
        st.markdown(f"**Answer:** {answer}")

    # ─── Display memory ──────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.subheader("🧠 Chat Memory (Persistent)")
        for i, (q, a) in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")
