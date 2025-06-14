# app.py  ──────────────────────────────────────────────────────────────
import streamlit as st
from uuid import uuid4
from typing import List, Tuple

from file_handlers import parse_pdf, parse_csv
from vectorstore_utils import prepare_vectorstore
from qa_chain import get_conversational_chain
from db_utils import init_db, save_chat, load_chat

# ─── Basic page config ────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("📄🔍 Groq-Powered Document Q&A Bot (Chat UI)")

openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
groq_api_key   = st.secrets.get("GROQ_API_KEY",   "")

# ─── Persistent session & DB init ─────────────────────────────────────
init_db()
session_id = st.session_state.get("session_id") or str(uuid4())
st.session_state["session_id"] = session_id

# chat_history will hold items like ("user", text) or ("assistant", text)
if "chat_history" not in st.session_state:
    raw_pairs: List[Tuple[str, str]] = load_chat(session_id)          # [(q, a)]
    st.session_state.chat_history = []
    for q, a in raw_pairs:
        st.session_state.chat_history.append(("user",      q))
        st.session_state.chat_history.append(("assistant", a))

# ─── File upload ──────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a CSV or PDF", type=["csv", "pdf"])

# Create the QA chain *after* we have an uploaded doc and API keys
if uploaded_file and openai_api_key and groq_api_key:

    # 1️⃣  Extract raw text from the file
    text = (
        parse_pdf(uploaded_file)
        if uploaded_file.name.lower().endswith(".pdf")
        else parse_csv(uploaded_file)
    )

    # 2️⃣  Embed + index; spin up ConversationalRetrievalChain
    with st.spinner("Embedding & indexing…"):
        vs       = prepare_vectorstore(text, openai_api_key)
        qa_chain = get_conversational_chain(vs, groq_api_key)
    st.success("Chatbot is ready!")

    # ─── Render existing history ABOVE the input box ──────────────────
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    # ─── Bottom-pinned input box ──────────────────────────────────────
    if prompt := st.chat_input("Ask something about your document"):

        # 1️⃣  Echo the user's message instantly
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2️⃣  Build chat_history in (q, a) tuples for the chain call
        linear_history = [
            (st.session_state.chat_history[i][1], st.session_state.chat_history[i + 1][1])
            for i in range(0, len(st.session_state.chat_history), 2)
            if i + 1 < len(st.session_state.chat_history)
        ]

        # 3️⃣  Invoke the conversational chain
        result  = qa_chain.invoke({"question": prompt, "chat_history": linear_history})
        answer  = result["answer"]

        # 4️⃣  Render assistant bubble
        with st.chat_message("assistant"):
            st.markdown(answer)

        # 5️⃣  Persist to session & DB
        st.session_state.chat_history.append(("user", prompt))
        st.session_state.chat_history.append(("assistant", answer))
        save_chat(session_id, prompt, answer)
