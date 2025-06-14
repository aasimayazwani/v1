# ======================= df_agent_utils.py =======================
from langchain_community.agent_toolkits.duckdb_sql import create_duckdb_sql_agent
from langchain_groq import ChatGroq
import duckdb


def build_sql_agent(df, groq_key, model="llama3-70b-8192"):
    llm = ChatGroq(groq_api_key=groq_key, model_name=model, temperature=0)
    con = duckdb.connect()  # in‑memory
    con.register("data", df)  # exposes as table "data"
    agent = create_duckdb_sql_agent(llm, con, prefix_messages=[
        "You can only execute read‑only SELECTs."
    ])
    return agent
