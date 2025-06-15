# ======================= app.py =======================
import streamlit as st
from uuid import uuid4
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from file_handlers import parse_pdf, parse_csv
from vectorstore_utils import prepare_vectorstore
from df_agent_utils import build_sql_agent, build_wikipedia_agent
from qa_chain import build_rag_chain
from db_utils import init_db, save_chat, load_chat

# ─── Page & secrets ────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("📄🔍 CSV & PDF Chatbot (SQL • RAG • Wikipedia)")

openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
groq_api_key   = st.secrets.get("GROQ_API_KEY", "")

if not (openai_api_key and groq_api_key):
    st.error("Add OPENAI_API_KEY and GROQ_API_KEY in `.streamlit/secrets.toml`.")
    st.stop()

# ─── Session & history ────────────────────────────────────────────────
init_db()
user_id = st.session_state.get("session_id") or str(uuid4())
st.session_state["session_id"] = user_id
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat(user_id)  # [(role,text)]

# ─── Sidebar: upload & opts ───────────────────────────────────────────
with st.sidebar:
    st.header("Upload document")
    uploaded_file = st.file_uploader("CSV or PDF", type=["csv", "pdf"])
    k_chunks = st.slider("Retriever top-k (RAG)", 4, 20, 12)
    show_src = st.checkbox("Show RAG source docs", value=False)

# ─── Always-on Wikipedia agent ────────────────────────────────────────
wiki_agent = build_wikipedia_agent(groq_api_key)

# ─── Build doc-dependent resources ────────────────────────────────────
df_agent  = None   # SQL/pandas agent
rag_chain = None

if uploaded_file:
    if uploaded_file.name.lower().endswith(".pdf"):
        docs = [parse_pdf(uploaded_file)]
    else:
        df, docs = parse_csv(uploaded_file)
        df_agent = build_sql_agent(df, groq_api_key)

    with st.spinner("Embedding & indexing …"):
        vs        = prepare_vectorstore(docs, openai_api_key)
        rag_chain = build_rag_chain(vs, groq_api_key, k_chunks)
    st.sidebar.success("Document indexed ✅")
else:
    st.sidebar.info("Chat with Wikipedia until you upload a file.")

# ─── Helper: detect “plot x vs y” and show chart ──────────────────────
def try_plot(prompt: str, agent) -> bool:
    """
    Very naive: if prompt contains 'plot x vs y', run SQL via agent,
    create a line plot, and display. Returns True if plotted.
    """
    tokens = prompt.lower().replace(",", " ").split()
    if "plot" in tokens and "vs" in tokens:
        try:
            x = tokens[tokens.index("plot") + 1]
            y = tokens[tokens.index("vs") + 1]
            query = f"SELECT {x}, {y} FROM data ORDER BY {x};"
            res = agent.run(query)
            if isinstance(res, pd.DataFrame) and {x, y}.issubset(res.columns):
                fig, ax = plt.subplots()
                ax.plot(res[x], res[y])
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                st.chat_message("assistant").pyplot(fig)
                return True
        except Exception:
            pass
    return False

# ─── Replay previous chat bubbles ─────────────────────────────────────
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)

# ─── Chat loop ────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question …"):
    st.chat_message("user").markdown(prompt)

    # 1️⃣ Wikipedia prefix
    if prompt.lower().startswith("wiki:"):
        answer = wiki_agent.run(prompt[5:].strip())
        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_history += [("user", prompt), ("assistant", answer)]
        save_chat(user_id, prompt, answer)
        st.stop()

    # 2️⃣ Try plotting helper if CSV + SQL agent exist
    if df_agent and try_plot(prompt, df_agent):
        st.session_state.chat_history += [("user", prompt), ("assistant", "(plot)")]
        save_chat(user_id, prompt, "(plot)")
        st.stop()

    # 3️⃣ Try SQL/pandas agent
    answer = None
    if df_agent:
        try:
            raw = df_agent.run(prompt)
            if isinstance(raw, pd.DataFrame):
                st.chat_message("assistant").dataframe(raw, use_container_width=True)
                answer = None  # visual only
            else:
                answer = str(raw)
        except Exception:
            answer = None

    # 4️⃣ Fallback to RAG
    if answer is None and rag_chain:
        tuples: List[Tuple[str, str]] = [
            (st.session_state.chat_history[i][1], st.session_state.chat_history[i + 1][1])
            for i in range(0, len(st.session_state.chat_history), 2)
            if i + 1 < len(st.session_state.chat_history)
        ]
        answer = rag_chain.invoke({"input": prompt, "chat_history": tuples})["answer"]

    # 5️⃣ If still none
    if answer is None:
        answer = "Sorry, I couldn't answer that."

    # Render answer text
    st.chat_message("assistant").markdown(answer)
    if show_src and isinstance(answer, dict) and "source_documents" in answer:
        with st.chat_message("assistant").expander("Sources"):
            st.write(answer["source_documents"])

    # Store history
    st.session_state.chat_history += [("user", prompt), ("assistant", answer)]
    save_chat(user_id, prompt, answer)
