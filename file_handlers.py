from PyPDF2 import PdfReader
import pandas as pd

def parse_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def parse_csv(file):
    df = pd.read_csv(file)
    return df.to_string(index=False)