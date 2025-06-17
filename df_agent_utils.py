"""
df_agent_utils.py
────────────────────────────────────────────────────────────────────────────
Helpers for creating LangChain agents that understand tabular data.

▸ What was here already?
    • build_dataframe_agent(llm, df, verbose=False) – wraps LangChain's
      `create_pandas_dataframe_agent`.

▸ What’s new?
    • CSVLookup (Tool)                               – ultra-light wrapper so
      each GTFS table can be queried safely.
    • make_agent(obj, llm, verbose=False)            – dispatcher that chooses
      between a single-DataFrame agent and a multi-table GTFS agent.

Nothing else in your codebase needs to change: wherever you used to call
`build_dataframe_agent`, you can now safely call `make_agent` instead.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from langchain.agents import (
    AgentType,
    create_pandas_dataframe_agent,
    initialize_agent,
)
from langchain.llms import OpenAI
from langchain.schema import BaseLanguageModel
from langchain.tools import Tool


# ───────────────────────────────────────────────────────────────────────────
# 1.  The original helper (kept intact for backwards compatibility)
# ───────────────────────────────────────────────────────────────────────────
def build_dataframe_agent(
    llm: BaseLanguageModel, df: pd.DataFrame, verbose: bool = False
):
    """
    Create a conversational agent that can run Python-level analysis on the
    supplied DataFrame.  This is identical to what you had before.
    """
    return create_pandas_dataframe_agent(
        llm,
        df,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
    )


# ───────────────────────────────────────────────────────────────────────────
# 2.  NEW – Very small tool for a single GTFS table
# ───────────────────────────────────────────────────────────────────────────
class CSVLookup(Tool):
    """
    A paper-thin LangChain Tool that lets an agent answer questions about
    one pandas DataFrame (e.g. `stops.txt`).

    You pass in the DataFrame at construction; the `_run` method then
    evaluates limited Python expressions against that DataFrame.
    """

    def __init__(self, table_name: str, df: pd.DataFrame):
        super().__init__()
        self.table_name = table_name
        self._df = df
        self.name = f"table_{table_name}"
        self.description = (
            f"Answer questions about GTFS table '{table_name}'. "
            "Call with a Python expression that references `df`."
        )

    # --- sync run ---------------------------------------------------------
    def _run(self, query: str) -> str:  # type: ignore[override]
        """
        Runs a *very* restricted `eval` so the agent can do things like:

            >>> tool.run('df["stop_id"].nunique()')

        This keeps the code surface minimal; tighten security as needed.
        """
        namespace: Dict[str, Any] = {"df": self._df, "pd": pd}
        try:
            result = eval(query, {"__builtins__": {}}, namespace)
            return str(result)
        except Exception as exc:
            return f"ERROR: {exc}"

    # --- async run (not used, so just raise) ------------------------------
    async def _arun(self, query: str) -> str:  # type: ignore[override]
        raise NotImplementedError("CSVLookup does not support async")


# ───────────────────────────────────────────────────────────────────────────
# 3.  NEW – Dispatcher that figures out which kind of agent to return
# ───────────────────────────────────────────────────────────────────────────
def make_agent(
    obj: Any, llm: BaseLanguageModel | None = None, verbose: bool = False
):
    """
    Factory that returns the *right* LangChain agent for the supplied object.

    Supported `obj` types
    ---------------------
    • `pandas.DataFrame`               –> same behaviour as before
    • `dict[str, pandas.DataFrame]`    –> treats each value as a GTFS table
    """
    # Fallback to OpenAI LLM if caller didn't supply one
    if llm is None:
        llm = OpenAI(temperature=0)

    # ---------------------------------------------------------------------
    # A) Single-table case  (just use the legacy helper)
    # ---------------------------------------------------------------------
    if isinstance(obj, pd.DataFrame):
        return build_dataframe_agent(llm, obj, verbose=verbose)

    # ---------------------------------------------------------------------
    # B) Multi-table case  (e.g. GTFS dict from _load_gtfs_zip)
    # ---------------------------------------------------------------------
    if isinstance(obj, dict):
        # Build one CSVLookup tool per table
        tools: List[Tool] = [
            CSVLookup(table_name=tbl_name, df=df) for tbl_name, df in obj.items()
        ]

        return initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=verbose,
        )

    # ---------------------------------------------------------------------
    # Fallback (unsupported type)
    # ---------------------------------------------------------------------
    raise TypeError(
        "make_agent only supports `pandas.DataFrame` or "
        "`dict[str, pandas.DataFrame]` objects."
    )
