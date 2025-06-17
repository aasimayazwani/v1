# ======================= app.py =======================
import io
import tempfile
from pathlib import Path
from typing import List, Tuple, Any

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from uuid import uuid4

# 🆕 New helpers that replaced the old ones
from file_handlers import ingest_any
from vectorstore_utils import VectorStore
from df_agent_utils import make_agent
from qa_chain import build_qa_chain, answer as rag_answer

from db_utils import init_db, save_chat, load_chat

# ---------------------------------------------------------------------
# 0.  Page config & API keys
# ---------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("📄🔍 Multi-File Chatbot (CSV • PDF • JSON • GTFS • Wikipedia)")

openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
groq_api_key   = st.secrets.get("GROQ_API_KEY", "")

if not (openai_api_key and groq_api_key):
    st.error(
        "Add OPENAI_API_KEY **and** GROQ_API_KEY in `.streamlit/secrets.toml` "
        "before running."
    )
    st.stop()

# ---------------------------------------------------------------------
# 1.  Tiny helper: LLM factory (Groq first, OpenAI fallback)
# ---------------------------------------------------------------------
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

def make_llm():
    try:
        return ChatGroq(groq_api_key=groq_api_key, model="mixtral-8x7b")
    except Exception:
        return ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

LLM = make_llm()      # reused everywhere to save rate-limits

# ---------------------------------------------------------------------
# 2.  Wikipedia agent (kept from old app but re-implemented)
# ---------------------------------------------------------------------
from langchain_community.tools import WikipediaQueryRun
from langchain.agents import AgentType, initialize_agent

def build_wikipedia_agent(llm):
    wiki_tool = WikipediaQueryRun()
    return initialize_agent(
        [wiki_tool],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )

wiki_agent = build_wikipedia_agent(LLM)

# ---------------------------------------------------------------------
# 3.  Session & SQLite chat history (unchanged)
# ---------------------------------------------------------------------
init_db()
user_id = st.session_state.get("session_id") or str(uuid4())
st.session_state["session_id"] = user_id
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat(user_id)  # [(role,text)]

# ---------------------------------------------------------------------
# 4.  Sidebar – file upload + opts
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Upload data file")
    uploaded_file = st.file_uploader(
        "CSV, PDF, JSON, or GTFS ZIP",
        type=["csv", "pdf", "json", "zip"],
    )
    k_chunks = st.slider("Retriever top-k (RAG)", 4, 20, 12)
    show_src = st.checkbox("Show RAG source docs", value=False)

# Variables that depend on the current upload
df_or_gtfs: Any = None          # DataFrame or dict[str,DF]
vectorstore: VectorStore | None = None

if uploaded_file:
    # Persist to a tmp path (LangChain needs a real file)
    tmp_path = Path(tempfile.mkdtemp()) / uploaded_file.name
    tmp_path.write_bytes(uploaded_file.getbuffer())

    # 4-A.  Ingest
    obj = ingest_any(tmp_path)

    # 4-B.  If it's a DataFrame (CSV / BusTime JSON)
    if isinstance(obj, pd.DataFrame) or isinstance(obj, dict):
        df_or_gtfs = obj
        agent = make_agent(df_or_gtfs, LLM)

    # 4-C.  If it's a list[Document] (PDF)
    if isinstance(obj, list):
        vectorstore = VectorStore("faiss")
        vectorstore.add_pdf_docs(obj)
        rag_chain = build_qa_chain(vectorstore, LLM)
    st.sidebar.success(f"Loaded & indexed: {uploaded_file.name} ✅")
else:
    st.sidebar.info("Chat with Wikipedia until you upload a file.")

# ---------------------------------------------------------------------
# 5.  Helper – naive "plot x vs y" recogniser (CSV DataFrame only)
# ---------------------------------------------------------------------
def try_plot(prompt: str, df_agent) -> bool:
    tokens = prompt.lower().replace(",", " ").split()
    if "plot" in tokens and "vs" in tokens:
        try:
            x = tokens[tokens.index("plot") + 1]
            y = tokens[tokens.index("vs") + 1]
            query = f"df[[{x!r}, {y!r}]].sort_values({x!r})"
            raw = df_agent.run(query)
            if isinstance(raw, pd.DataFrame) and {x, y}.issubset(raw.columns):
                fig, ax = plt.subplots()
                ax.plot(raw[x], raw[y])
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                st.chat_message("assistant").pyplot(fig)
                return True
        except Exception:
            pass
    return False

# ---------------------------------------------------------------------
# 6.  Replay previous chat bubbles
# ---------------------------------------------------------------------
for role, msg in st.session_state.chat_history:
    if role == "assistant" and msg == "(plot)":
        # Skip re-rendering old plots
        st.chat_message("assistant").markdown("*(plot)*")
    else:
        st.chat_message(role).markdown(msg)

# ---------------------------------------------------------------------
# 7.  Chat loop
# ---------------------------------------------------------------------
if prompt := st.chat_input("Ask a question …"):
    st.chat_message("user").markdown(prompt)

    # 7-A.  Wikipedia shortcut
    if prompt.lower().startswith("wiki:"):
        answer = wiki_agent.run(prompt[5:].strip())
        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_history += [("user", prompt), ("assistant", answer)]
        save_chat(user_id, prompt, answer)
        st.stop()

    # 7-B.  CSV / GTFS agent (includes plot helper)
    if isinstance(df_or_gtfs, (pd.DataFrame, dict)):
        if try_plot(prompt, agent):
            st.session_state.chat_history += [("user", prompt), ("assistant", "(plot)")]
            save_chat(user_id, prompt, "(plot)")
            st.stop()
        try:
            raw = agent.run(prompt)
            if isinstance(raw, pd.DataFrame):
                st.chat_message("assistant").dataframe(raw, use_container_width=True)
                answer = ""
            else:
                answer = str(raw)
        except Exception:
            answer = None
    else:
        answer = None

    # 7-C.  RAG fallback (PDF or when agent failed)
    if answer in (None, "") and vectorstore is not None:
        answer_text, src_docs = rag_answer(prompt, vectorstore, LLM)
        answer = answer_text
        if show_src:
            with st.expander("🔎 RAG Sources"):
                for d in src_docs:
                    st.write(f"• {d.metadata}")

    # 7-D.  Final fallback to Wikipedia
    if answer in (None, ""):
        answer = wiki_agent.run(prompt)

    # 7-E.  Render & persist
    st.chat_message("assistant").markdown(answer)
    st.session_state.chat_history += [("user", prompt), ("assistant", answer)]
    save_chat(user_id, prompt, answer)
