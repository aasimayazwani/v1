"""Utility builders for analytic (DuckDB **or** pandas) and Wikipedia agents."""
from typing import List
import duckdb
import pandas as pd
from langchain_groq import ChatGroq

# ── Try DuckDB agent first; fall back to pandas agent if missing ─────────────
try:
    # Newer path (may exist in some versions)
    from langchain_experimental.agent_toolkits.duckdb_sql import create_duckdb_agent
    _DUCKDB_OK = True
except ImportError:
    try:
        # Older path
        from langchain_experimental.agent_toolkits import create_duckdb_agent  # type: ignore
        _DUCKDB_OK = True
    except ImportError:
        _DUCKDB_OK = False
        from langchain_experimental.agents import create_pandas_dataframe_agent

# ── Wikipedia tool imports ───────────────────────────────────────────────────
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType

# ---------------------------------------------------------------------------
# Analytics agent builder (DuckDB if available, else pandas DataFrame agent)
# ---------------------------------------------------------------------------
def build_sql_agent(df: pd.DataFrame, groq_key: str, model: str = "llama3-70b-8192"):
    """Return an analytic agent for the given DataFrame.

    *Prefers* DuckDB read-only SQL. Falls back to pandas-repl agent if
    DuckDB toolkit isn’t installed.
    """
    llm = ChatGroq(groq_api_key=groq_key, model_name=model, temperature=0)

    if _DUCKDB_OK:
        con = duckdb.connect()
        con.register("data", df)
        return create_duckdb_agent(
            llm=llm,
            db=con,
            include_dataframe_in_prompt=True,
            prefix_messages=["You are a data analyst. Only run read-only SELECTs."],
        )

    # Fallback: pandas agent (requires allow_dangerous_code=True)
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,
    )

# ---------------------------------------------------------------------------
# Wikipedia zero-shot REACT agent
# ---------------------------------------------------------------------------
def build_wikipedia_agent(groq_key: str, model: str = "llama3-70b-8192"):
    llm = ChatGroq(groq_api_key=groq_key, model_name=model, temperature=0)
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return initialize_agent(
        [wiki_tool],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )
