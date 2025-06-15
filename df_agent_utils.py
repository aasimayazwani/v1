# ======================= df_agent_utils.py =======================
"""
Utility builders:

1) build_sql_agent(df, groq_key, model=…)  ->  DuckDB SQL agent  
   • Registers the DataFrame as table `data`.  
   • If the DuckDB toolkit isn’t installed, falls back to a pandas agent.

2) build_wikipedia_agent(groq_key, model=…) -> Wikipedia REACT agent
"""

from typing import Any

import duckdb
import pandas as pd
from langchain_groq import ChatGroq

# ── Robust import for DuckDB toolkit ───────────────────────────────
try:  # newest path (≥ 0.0.50)
    from langchain_experimental.agent_toolkits.duckdb_sql import create_duckdb_agent
except ImportError:
    try:  # older path
        from langchain_experimental.agent_toolkits import create_duckdb_agent
    except ImportError:
        create_duckdb_agent = None  # triggers pandas fallback

# Fallback pandas DataFrame agent (always available)
from langchain_experimental.agents import create_pandas_dataframe_agent

# Wikipedia tool imports (LangChain ≥ 0.2)
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType


# ───────────────────────────────────────────────────────────────────
def build_sql_agent(
    df: pd.DataFrame,
    groq_key: str,
    model: str = "llama3-70b-8192",
) -> Any:
    """Return a read-only SQL/analytics agent for the DataFrame."""
    llm = ChatGroq(groq_api_key=groq_key, model_name=model, temperature=0)

    # Preferred: DuckDB agent (safe, schema-aware)
    if create_duckdb_agent:
        con = duckdb.connect()
        con.register("data", df)  # SQL table name = data
        return create_duckdb_agent(
            llm=llm,
            db=con,
            include_dataframe_in_prompt=True,
            prefix_messages=[
                "You are a data analyst. Use only read-only SELECT statements."
            ],
        )

    # Fallback: pandas agent (requires allow_dangerous_code)
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,
    )


def build_wikipedia_agent(
    groq_key: str,
    model: str = "llama3-70b-8192",
):
    """Zero-shot REACT agent with the Wikipedia tool."""
    llm = ChatGroq(groq_api_key=groq_key, model_name=model, temperature=0)
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    return initialize_agent(
        tools=[wiki_tool],
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )
