"""Utility builders for analytic (DuckDB) and external-tool agents."""
from langchain_groq import ChatGroq
from langchain_experimental.agent_toolkits import create_duckdb_agent
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType
import duckdb
import pandas as pd
from typing import List


# ─── DuckDB SQL AGENT ──────────────────────────────────────────────────────
def build_sql_agent(
    df: pd.DataFrame,
    groq_key: str,
    model: str = "llama3-70b-8192",
):
    """Expose DataFrame as table `data` and return a read-only SQL agent."""
    llm = ChatGroq(groq_api_key=groq_key, model_name=model, temperature=0)

    con = duckdb.connect()
    con.register("data", df)  # table name = data

    agent = create_duckdb_agent(
        llm=llm,
        db=con,
        include_dataframe_in_prompt=True,
        prefix_messages=[
            "You are a data analyst. Only run read-only SELECT statements."
        ],
    )
    return agent


# ─── Wikipedia TOOL AGENT ──────────────────────────────────────────────────
def build_wikipedia_agent(
    groq_key: str,
    model: str = "llama3-70b-8192",
):
    """Zero-shot REACT agent that can call the Wikipedia tool."""
    llm = ChatGroq(groq_api_key=groq_key, model_name=model, temperature=0)
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools: List = [wiki_tool]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )
    return agent
