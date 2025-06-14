"""Analytics (DuckDB → SQL) and Wikipedia tool agents with graceful fallback."""
from typing import List
import pandas as pd, duckdb
from langchain_groq import ChatGroq

# --- try DuckDB toolkits; else fall back to pandas agent -----------------
try:
    from langchain_experimental.agents.agent_toolkits.duckdb_sql import create_duckdb_agent
    _HAS_DUCK = True
except ImportError:
    try:
        from langchain_experimental.agent_toolkits import create_duckdb_agent  # <0.0.55
        _HAS_DUCK = True
    except ImportError:
        _HAS_DUCK = False
        from langchain_experimental.agents import create_pandas_dataframe_agent

# --- Wikipedia tool ------------------------------------------------------
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType


def build_sql_agent(df: pd.DataFrame, groq_key: str, model="llama3-70b-8192"):
    llm = ChatGroq(groq_api_key=groq_key, model_name=model, temperature=0)

    if _HAS_DUCK:
        con = duckdb.connect()
        con.register("data", df)
        return create_duckdb_agent(
            llm=llm,
            db=con,
            include_dataframe_in_prompt=True,
            prefix_messages=["Use read-only SELECT statements only."],
        )

    # fallback: pandas agent (less safe)
    return create_pandas_dataframe_agent(
        llm, df, verbose=False, allow_dangerous_code=True
    )


def build_wikipedia_agent(groq_key: str, model="llama3-70b-8192"):
    llm = ChatGroq(groq_api_key=groq_key, model_name=model, temperature=0)
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return initialize_agent([wiki_tool], llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION)
