"""Utility builders for analytic (DuckDB) and external-tool agents."""

from langchain_groq import ChatGroq
import duckdb
import pandas as pd
from typing import List

# ── DuckDB agent import (works for ≥0.0.50) ─────────────────────────────
try:
    # Newer path
    from langchain_experimental.agents.agent_toolkits.duckdb_sql import (
        create_duckdb_agent,
    )
except ImportError:  # fallback for slightly older 0.0.5x versions
    from langchain_experimental.agent_toolkits import create_duckdb_agent

# ── Wikipedia tool imports ───────────────────────────────────────────────
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType


# ───────────────── DuckDB SQL agent ──────────────────────────────────────
def build_sql_agent(
    df: pd.DataFrame,
    groq_key: str,
    model: str = "llama3-70b-8192",
):
    """Expose DataFrame as table `data` and return read-only SQL agent."""
    llm = ChatGroq(groq_api_key=groq_key, model_name=model, temperature=0)
    con = duckdb.connect()
    con.register("data", df)  # table name: data

    return create_duckdb_agent(
        llm=llm,
        db=con,
        include_dataframe_in_prompt=True,
        prefix_messages=["You are a data analyst. Only run read-only SELECTs."],
    )


# ───────────────── Wikipedia tool agent ───────────────────────────────────
def build_wikipedia_agent(
    groq_key: str,
    model: str = "llama3-70b-8192",
):
    """Zero-shot REACT agent that can call Wikipedia."""
    llm = ChatGroq(groq_api_key=groq_key, model_name=model, temperature=0)
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools: List = [wiki_tool]

    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )
