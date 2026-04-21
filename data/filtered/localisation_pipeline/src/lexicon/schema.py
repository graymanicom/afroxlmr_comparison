from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import load_yaml


@dataclass(frozen=True)
class LexiconEntry:
    surface: str
    lang: str
    kind: str
    semantic_type: str
    local_marker: bool
    provenance: str = "seed"
    score: float = 1.0
    count: int = 1


def load_seed_lexicon(path: str | Path) -> pd.DataFrame:
    raw = load_yaml(path)
    entries = raw.get("entries", [])
    df = pd.DataFrame(entries)
    if df.empty:
        raise ValueError(f"Seed lexicon at {path} is empty")
    df["surface"] = df["surface"].str.casefold()
    df["provenance"] = "seed"
    df["score"] = 1.0
    df["count"] = 1
    return df[
        ["surface", "lang", "kind", "semantic_type", "local_marker", "provenance", "score", "count"]
    ].copy()
