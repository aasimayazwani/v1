# file_handlers.py
import pandas as pd
from langchain.docstore.document import Document


def parse_csv(file):
    """
    Returns
    -------
    df      : pandas.DataFrame            (kept in memory for analytics)
    docs    : list[Document] — one per row (for semantic retrieval)
    """
    df = pd.read_csv(file)
    docs = [
        Document(
            page_content=" | ".join(f"{col}: {val}" for col, val in row.items()),
            metadata={"row_index": int(i)},
        )
        for i, row in df.iterrows()
    ]
    return df, docs


def parse_pdf(file):
    from PyPDF2 import PdfReader
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)