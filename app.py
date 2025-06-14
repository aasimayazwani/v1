# app.py  ──────────────────────────────────────────────────────────────
import streamlit as st
from uuid import uuid4
from typing import List, Tuple

from file_handlers import parse_pdf, parse_csv           # parse_csv now returns (df, docs)
from vectorstore_utils import prepare_vectorstore        # accepts list[Document]
from qa_chain import get_conversational_chain            # now has search_k param
from df_agent_utils import build_df_agent                # new helper for analytics path
from db_utils import init_db, save_chat, load_chat

# ─── Page config ──────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("📄🔍 Groq-Powered CSV & PDF Chatbot")

openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
groq_api_key   = st.secrets.get("GROQ_API_KEY",   "")

# ─── Persistent session & DB init ─────────────────────────────────────
init_db()
session_id = st.session_state.get("session_id") or str(uuid4())
st.session_state["session_id"] = session_id

# Store bubbles as ("user"| "assistant", text)
if "chat_history" not in st.session_state:
    raw_pairs: List[Tuple[str, str]] = load_chat(session_id)        # [(q, a)]
    st.session_state.chat_history = []
    for q, a in raw_pairs:
        st.session_state.chat_history += [("user", q), ("assistant", a)]

# ─── File upload ──────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a CSV or PDF", type=["csv", "pdf"])

if uploaded_file and openai_api_key and groq_api_key:

    # 1️⃣  Parse file ---------------------------------------------------
    if uploaded_file.name.lower().endswith(".pdf"):
        raw_text  = parse_pdf(uploaded_file)
        df        = None
        docs      = [raw_text]                  # single document string
    else:
        df, docs  = parse_csv(uploaded_file)    # df for analytics, docs list for vectors

    # 2️⃣  Vectorstore & chains ----------------------------------------
    with st.spinner("Embedding & indexing…"):
        vs        = prepare_vectorstore(docs, openai_api_key)
        qa_chain  = get_conversational_chain(vs, groq_api_key, search_k=12)
        if df is not None:
            df_agent = build_df_agent(df, groq_api_key)             # Pandas agent

    st.success("Chatbot is ready!")

    # ─── Render old messages before input (chat bubbles) --------------
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # ─── Chat input (always sticks to bottom) -------------------------
    if prompt := st.chat_input("Ask something about your document"):

        # 1️⃣  Show user bubble
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2️⃣  Build (q,a) tuples for chain
        linear_hist = [
            (st.session_state.chat_history[i][1], st.session_state.chat_history[i + 1][1])
            for i in range(0, len(st.session_state.chat_history), 2)
            if i + 1 < len(st.session_state.chat_history)
        ]

        # 3️⃣  Simple heuristic: words that imply math/aggregation
        analytic_keywords = (
            "average", "mean", "sum", "count", "total", "minimum",
            "maximum", "median", "std", "standard deviation", "plot",
            "histogram", "trend"
        )
        analytic = any(k in prompt.lower() for k in analytic_keywords)

        # 4️⃣  Route the query
        answer: str | None = None

        if analytic and df is not None:
            try:
                answer = df_agent.run(prompt)
            except Exception:
                answer = None  # fallback to semantic if pandas fails

        if answer is None:
            result  = qa_chain.invoke({"question": prompt, "chat_history": linear_hist})
            answer  = result["answer"]

            # optional fallback: if LLM says "I don't know" and we have df
            if ("i don’t know" in answer.lower() or "idk" in answer.lower()) and df is not None:
                try:
                    answer = df_agent.run(prompt)
                except Exception:
                    pass

        # 5️⃣  Show assistant bubble
        with st.chat_message("assistant"):
            st.markdown(answer)

        # 6️⃣  Persist to session + DB
        st.session_state.chat_history += [("user", prompt), ("assistant", answer)]
        save_chat(session_id, prompt, answer)
