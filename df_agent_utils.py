# df_agent_utils.py
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq


def build_df_agent(df, groq_api_key, model_name="llama3-70b-8192"):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0)
    return create_pandas_dataframe_agent(llm, df, verbose=False)
