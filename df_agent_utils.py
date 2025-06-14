# ======================= df_agent_utils.py =======================
from langchain_experimental.agent_toolkits import create_duckdb_agent
from langchain_groq import ChatGroq
import duckdb


def build_sql_agent(df, groq_key, model="llama3-70b-8192"):
    """
    Builds a SQL agent using DuckDB to query the in-memory DataFrame.
    Relies on LangChain Experimental's `create_duckdb_agent`.

    Args:
        df (pd.DataFrame): Input DataFrame to expose as SQL table.
        groq_key (str): Your Groq API key.
        model (str): LLM model name, defaults to "llama3-70b-8192".

    Returns:
        LangChain Agent: A SQL-capable agent for use in RAG/Q&A flows.
    """
    llm = ChatGroq(groq_api_key=groq_key, model_name=model, temperature=0)

    # DuckDB in-memory database and register the DataFrame
    con = duckdb.connect()
    con.register("data", df)  # expose the DataFrame as a SQL table called "data"

    # Create the DuckDB agent
    agent = create_duckdb_agent(llm=llm, db=con, include_dataframe_in_prompt=True)
    return agent
