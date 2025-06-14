# app.py
import streamlit as st
import uuid

from file_handlers import parse_pdf, parse_csv
from vectorstore_utils import prepare_vectorstore
from qa_chain import get_groq_chain_with_history
from db_utils import init_db, save_chat, load_chat

st.set_page_config(layout="wide")
st.title("📄🔍 Groq-Powered Document Q&A Bot")

openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
groq_api_key   = st.secrets.get("GROQ_API_KEY",   "")

# ─── Initialise persistent DB + session ──────────────────────────────────────
init_db()
session_id = st.session_state.get("session_id") or str(uuid.uuid4())
st.session_state["session_id"] = session_id
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat(session_id)

# ─── File upload ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a CSV or PDF", type=["csv", "pdf"])

if uploaded_file and openai_api_key and groq_api_key:
    # Extract raw text
    text = (parse_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf")
            else parse_csv(uploaded_file))

    # Build vectorstore + QA chain
    with st.spinner("Embedding & indexing…"):
        vs        = prepare_vectorstore(text, openai_api_key)
        qa_chain  = get_groq_chain_with_history(vs, groq_api_key)
    st.success("Chatbot is ready!")

    # ─── Chat loop ───────────────────────────────────────────────────────────
    query = st.text_input("Ask something about your document:")
    if query:
        # Format memory
        history_text = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history
        ) or "No prior conversation."

        # Call RetrievalQA  (only 'query' + any extra vars)
        inputs  = {"query": query.strip(), "history": history_text}
        answer  = qa_chain.invoke(inputs)["result"]   # invoke() is the safe entry point

        # Persist + display
        st.session_state.chat_history.append((query, answer))
        save_chat(session_id, query, answer)
        st.markdown(f"**Answer:** {answer}")

    # ─── Display running memory ──────────────────────────────────────────────
    if st.session_state.chat_history:
        st.subheader("🧠 Chat Memory (Persistent)")
        for i, (q, a) in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")
