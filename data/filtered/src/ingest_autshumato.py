from pathlib import Path
from typing import Iterable
import pandas as pd

def ingest_autshumato(en_file: Path, xx_file: Path, xx_iso3: str, source: str="autshumato") -> pd.DataFrame:
    en_lines = en_file.read_text(encoding="utf-8").splitlines()
    xx_lines = xx_file.read_text(encoding="utf-8").splitlines()
    if len(en_lines) != len(xx_lines):
        raise ValueError(f"Alignment mismatch: {len(en_lines)} vs {len(xx_lines)}")
    rows = []
    for i, (en, xx) in enumerate(zip(en_lines, xx_lines)):
        en = en.strip()
        xx = xx.strip()
        if not en or not xx:
            continue
        rows.append({
            "source": source,
            "doc_id": f"{en_file.stem}__{xx_iso3}",
            "sent_id": f"{en_file.stem}__{xx_iso3}__{i}",
            "language": "eng",
            "text_raw": en,
        })
        rows.append({
            "source": source,
            "doc_id": f"{en_file.stem}__{xx_iso3}",
            "sent_id": f"{en_file.stem}__{xx_iso3}__{i}",
            "language": xx_iso3,
            "text_raw": xx,
        })
    return pd.DataFrame(rows)