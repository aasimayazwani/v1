# ======================= df_agent_utils.py =======================
"""
Builds a read-only DuckDB SQL agent so the chatbot can answer
questions like:

    • Which bus_id has the lowest avg_kwh_mile?
    • SELECT avg(total_dist_block) FROM data WHERE weekday(date)=3 GROUP BY bus_id;

Table name exposed to the agent is **data**.
"""
import duckdb
import pandas as pd
from typing import Any

from langchain_groq import ChatGroq
# DuckDB toolkit lives here for langchain-experimental ≥ 0.0.50
from langchain_experimental.agent_toolkits.duckdb_sql import create_duckdb_agent


def build_sql_agent(df: pd.DataFrame, groq_key: str) -> Any:
    """Return a DuckDB SQL agent wired to the DataFrame."""
    llm = ChatGroq(groq_api_key=groq_key,
                   model_name="llama3-70b-8192",
                   temperature=0)

    con = duckdb.connect()
    con.register("data", df)          # table name that SQL queries see

    # include_dataframe_in_prompt=True gives schema preview to the LLM
    return create_duckdb_agent(
        llm=llm,
        db=con,
        include_dataframe_in_prompt=True,
        prefix_messages=["You are a data analyst. Use only read-only SELECT statements."],
    )
