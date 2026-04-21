from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import AutshumatoConfig
from src.utils.text import normalize_text


def ingest_autshumato(config: AutshumatoConfig) -> pd.DataFrame:
    """Read sentence-aligned English/non-English text files from local disk.

    Output columns follow the pair-oriented design you requested:
    - english
    - other_sentence
    - language
    - source

    Additional provenance columns are included because they are useful later
    for debugging, auditing, and lexicon induction.
    """
    rows: list[dict[str, object]] = []

    for pair in config.pairs:
        english_path = Path(pair.english_path)
        other_path = Path(pair.other_path)

        if not english_path.exists():
            raise FileNotFoundError(f"Autshumato English file not found: {english_path}")
        if not other_path.exists():
            raise FileNotFoundError(f"Autshumato target file not found: {other_path}")

        english_lines = english_path.read_text(encoding="utf-8").splitlines()
        other_lines = other_path.read_text(encoding="utf-8").splitlines()

        if len(english_lines) != len(other_lines):
            raise ValueError(
                "Alignment mismatch for Autshumato pair "
                f"{english_path.name} vs {other_path.name}: "
                f"{len(english_lines)} != {len(other_lines)}"
            )

        for idx, (eng, other) in enumerate(zip(english_lines, other_lines)):
            eng = normalize_text(eng)
            other = normalize_text(other)
            if not eng or not other:
                continue
            rows.append(
                {
                    "pair_id": f"autshumato::{pair.language}::{english_path.stem}::{idx}",
                    "english": eng,
                    "other_sentence": other,
                    "language": pair.language,
                    "source": "autshumato",
                    "source_doc_id": english_path.stem,
                    "source_row_id": idx,
                }
            )

    return pd.DataFrame(rows)
