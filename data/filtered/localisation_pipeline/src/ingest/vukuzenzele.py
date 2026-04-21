from __future__ import annotations
from typing import Any
import pandas as pd
from src.config import VukuzenzeleConfig
from src.utils.text import normalize_text
import os
from datasets import load_dataset

try: 
    HF_TOKEN = os.getenv("HF_TOKEN")
except:   
    HF_TOKEN = None
    print("No Hugging Face token found in environment variable HF_TOKEN. If you are trying to load a private dataset, please set this variable and try again.")

def load_dataset_with_auth(name: str, subset: str | None = None, **kwargs):
    return load_dataset(
        name,
        subset,
        token=HF_TOKEN,
        **kwargs,
    )

def _subset_languages(subset_name: str) -> tuple[str, str]:
    left, right = subset_name.split("-", maxsplit=1)
    return left, right


def ingest_vukuzenzele(config: VukuzenzeleConfig) -> pd.DataFrame:
    """Load sentence-aligned Vuk'uzenzele pairs from Hugging Face.

    We only keep subsets that include English, because the downstream filter is run on English first.
    Each row represents a single aligned pair.
    """
    rows: list[dict[str, Any]] = []

    for subset in config.subsets:
        left_lang, right_lang = _subset_languages(subset)
        if "eng" not in {left_lang, right_lang}:
            raise ValueError(
                f"Subset {subset} does not include English. "
                "Use only English-including subsets for this pipeline."
            )

        split_expr = "+".join(config.splits)
        ds = load_dataset_with_auth(config.repo_id, subset, split=split_expr)

        for row_idx, record in enumerate(ds):
            score = float(record.get("score", 1.0))
            if score < config.min_alignment_score:
                continue

            eng_text = normalize_text(record["eng"])
            other_lang = right_lang if left_lang == "eng" else left_lang
            other_col = other_lang
            other_text = normalize_text(record[other_col])

            if not eng_text or not other_text:
                continue

            rows.append(
                {
                    "pair_id": f"vukuzenzele_hf::{subset}::{row_idx}",
                    "english": eng_text,
                    "other_sentence": other_text,
                    "language": other_lang,
                    "source": "vukuzenzele_hf",
                    "source_doc_id": subset,
                    "source_row_id": row_idx,
                    "alignment_score": score,
                }
            )

    return pd.DataFrame(rows)
