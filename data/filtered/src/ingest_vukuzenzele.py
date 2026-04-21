from pathlib import Path
import pandas as pd
import pdfplumber

def extract_pdf_text(pdf_path: Path) -> str:
    parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)
    
def ingest_vukuzenzele_pdf(pdf_path: Path, lang_iso3: str, edition_id: str, source: str="vukuzenzele") -> pd.DataFrame:
    raw = extract_pdf_text(pdf_path)
    return pd.DataFrame([{
        "source": source,
        "doc_id": f"{edition_id}__{lang_iso3}",
        "sent_id": f"{edition_id}__{lang_iso3}__doc",
        "language": lang_iso3,
        "text_raw": raw,
    }])