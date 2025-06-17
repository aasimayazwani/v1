"""
file_handlers.py
────────────────────────────────────────────────────────────────────────────
Lightweight helpers for loading user-supplied data files (CSV, JSON, PDF,
and GTFS ZIP) so downstream LangChain chains or dataframe agents can
consume them.

✓  Keeps all original CSV / JSON helpers.
✓  Adds:
      • _load_pdf               → reads text from multi-page PDFs
      • _load_vehicles_json     → parses BusTime getvehicles snapshots
      • _load_gtfs_zip          → extracts primary GTFS tables to DataFrames
      • ingest_any              → single entry-point; returns objects suited
                                   to your existing vector / agent pipeline

All new code is clearly marked; existing functions are unchanged.
"""

from __future__ import annotations

import json
import io
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Union

import pandas as pd
import magic  # python-magic for MIME sniffing
from pypdf import PdfReader
from unidecode import unidecode
from langchain.schema import Document

# ───────────────────────────────────────────────────────────────────────────
# ORIGINAL HELPERS  (kept exactly as before)
# ───────────────────────────────────────────────────────────────────────────

def load_csv(
    path: str | Path,
    sep: str = ",",
    encoding: str = "utf-8",
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Basic CSV loader used elsewhere in the codebase.

    Parameters
    ----------
    path : str | Path
        Absolute or relative path on disk.
    sep : str
        Field delimiter.
    encoding : str
        Text encoding.
    read_csv_kwargs : dict
        Extra args passed straight to `pd.read_csv`.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path, sep=sep, encoding=encoding, **read_csv_kwargs)


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Generic JSON loader (kept for backwards compatibility).

    Returns the raw dict so callers can decide how to normalise.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ───────────────────────────────────────────────────────────────────────────
# NEW HELPERS  (PDF  ▸  BusTime JSON  ▸  GTFS ZIP)
# ───────────────────────────────────────────────────────────────────────────

def _load_pdf(path: str | Path) -> List[Document]:
    """
    Extract plain text from every page in a PDF and wrap it in LangChain
    `Document` objects so the text chunks are retrieval-friendly.

    • Unicode is normalised with `unidecode` for cleaner embeddings.
    • Page number is stored in metadata for source tracking.
    """
    reader = PdfReader(str(path))
    docs: List[Document] = []
    for page_idx, page in enumerate(reader.pages):
        text: str = page.extract_text() or ""
        text = unidecode(text)
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path), "page": page_idx + 1},
            )
        )
    return docs


def _load_vehicles_json(path: str | Path) -> pd.DataFrame:
    """
    Parse an MTA BusTime `getvehicles` JSON snapshot.

    The structure is:
        {
          "bustime-response": {
              "vehicle": [ { ... }, ... ]
          }
        }
    The returned DataFrame is type-cast so numeric columns are genuine floats.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = raw["bustime-response"]["vehicle"]
    df = pd.DataFrame(data)

    # Coerce commonly numeric fields
    for col in ("lat", "lon", "spd", "rtdir"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


_PRIMARY_GTFS = [
    # Minimal core tables; extend list if you need shapes.txt, alerts, etc.
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "agency.txt",
]


def _load_gtfs_zip(path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Decompress a GTFS feed (.zip) and read the primary text tables into a
    dict of DataFrames keyed by the original filename.

    Any missing optional table is silently skipped.
    """
    dfs: Dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as z:
        for filename in _PRIMARY_GTFS:
            if filename not in z.namelist():
                continue
            with z.open(filename) as file_handle:
                dfs[filename] = pd.read_csv(
                    io.TextIOWrapper(file_handle, encoding="utf-8")
                )
    return dfs


# ───────────────────────────────────────────────────────────────────────────
# UNIFIED ENTRY-POINT
# ───────────────────────────────────────────────────────────────────────────

def ingest_any(
    path: str | Path,
) -> Union[List[Document], pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Auto-detect file type (via libmagic) and delegate to the appropriate
    loader.  Returns one of three object types:

      • list[Document]                 → for PDFs
      • pd.DataFrame                   → for CSV or BusTime JSON
      • dict[str, pd.DataFrame]        → for GTFS ZIP feeds

    Callers can inspect the return type and route to either:
      * vectorstore_utils.add_pdf_docs()
      * dataframe agents
      * custom GTFS tools
    """
    mime: str = magic.from_file(str(path), mime=True)

    # PDF
    if mime == "application/pdf":
        return _load_pdf(path)

    # JSON (BusTime)
    if mime == "application/json":
        # Heuristic: if the top-level key matches BusTime, use the custom
        # loader; otherwise just return raw JSON.
        with open(path, "r", encoding="utf-8") as f:
            first_token = f.readline(256)
        if '"bustime-response"' in first_token:
            return _load_vehicles_json(path)
        return load_json(path)

    # GTFS feed (.zip)
    if mime in ("application/zip", "application/x-zip-compressed"):
        return _load_gtfs_zip(path)

    # CSV or plain text → use existing loader
    if mime in ("text/csv", "text/plain"):
        return load_csv(path)

    raise ValueError(f"Unsupported or unrecognised file type: {mime}")
