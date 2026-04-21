from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd
from src.config import LexiconConfig
from src.utils.text import count_ngrams, score_candidates, tokenize

import re

_WORD_RE = re.compile(r"^\w+$", re.UNICODE)

@dataclass(frozen=True)
class InductionStats:
    source: str
    seed_surface: str
    seed_kind: str
    seed_semantic_type: str
    language: str
    seed_sentence_count: int
    candidate_count: int

def _token_count(text: str) -> int:
    return len([t for t in str(text).split() if t.strip()])

def _passes_kind_length(surface: str, kind: str, prune_cfg) -> bool:
    n = _token_count(surface)
    min_n = prune_cfg.min_tokens_by_kind.get(kind, 1)
    max_n = prune_cfg.max_tokens_by_kind.get(kind, 99)
    return min_n <= n <= max_n


def _passes_edge_stopword(surface: str, stopwords: set[str], prune_cfg) -> bool:
    toks = [t for t in str(surface).split() if t.strip()]
    if not toks:
        return False
    if prune_cfg.drop_leading_stopword and toks[0] in stopwords:
        return False
    if prune_cfg.drop_trailing_stopword and toks[-1] in stopwords:
        return False
    return True

def _apply_candidate_shape_filters(
    candidate: str,
    kind: str,
    stopwords: set[str],
    prune_cfg,
) -> bool:
    if not _passes_kind_length(candidate, kind, prune_cfg):
        return False
    if not _passes_edge_stopword(candidate, stopwords, prune_cfg):
        return False
    return True

def prune_subphrases(
    df: pd.DataFrame,
    subphrase_score_ratio: float = 0.9,
) -> pd.DataFrame:
    """
    Keep longer, stronger phrases and suppress weaker strict subphrases
    within the same (lang, kind, semantic_type, seed_surface, source).
    """
    if df.empty:
        return df.copy()

    work = df.copy()
    work["token_len"] = work["surface"].map(_token_count)

    group_cols = ["induced_from_source", "seed_surface", "lang", "kind", "semantic_type"]
    kept_groups = []

    for _, group in work.groupby(group_cols, dropna=False):
        group = group.sort_values(
            by=["score", "count", "token_len"],
            ascending=[False, False, False],
        )
        kept = []

        for row in group.to_dict("records"):
            surf = row["surface"]
            row_score = float(row["score"])
            shadowed = False

            for prev in kept:
                prev_surf = prev["surface"]
                prev_score = float(prev["score"])
                if surf != prev_surf and surf in prev_surf:
                    if row_score <= prev_score * subphrase_score_ratio:
                        shadowed = True
                        break

            if not shadowed:
                kept.append(row)

        kept_groups.append(pd.DataFrame(kept))

    out = pd.concat(kept_groups, ignore_index=True) if kept_groups else work.iloc[0:0].copy()
    return out.drop(columns=["token_len"], errors="ignore")

def _load_stopwords(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stopword file not found: {path}")
    return {line.strip().casefold() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


import re


_WORD_RE = re.compile(r"^\w+$", re.UNICODE)


def _valid_candidate(candidate: str, min_token_chars: int, stopwords: set[str],) -> bool:
    """
    Decide whether an n-gram candidate is worth keeping before scoring.

    This is a *high-recall but structured* filter:
    - removes obvious junk (stopwords, punctuation, digits)
    - enforces minimal lexical content
    - keeps plausible multiword institutional phrases
    """

    if not candidate:
        return False

    candidate = candidate.strip().lower()
    tokens = [t for t in candidate.split() if t.strip()]

    if not tokens:
        return False

    # --- 1. Length / token sanity ---
    if any(len(t) < min_token_chars for t in tokens):
        return False

    # --- 2. Reject non-word tokens ---
    # (keeps things like "binnelandse", rejects punctuation fragments)
    if not all(_WORD_RE.match(t) for t in tokens):
        return False

    # --- 3. Reject numeric-heavy tokens ---
    if any(any(c.isdigit() for c in t) for t in tokens):
        return False

    # --- 4. Reject all-stopword candidates ---
    if all(t in stopwords for t in tokens):
        return False

    # --- 5. Require at least one content token ---
    # (i.e. not stopword)
    content_tokens = [t for t in tokens if t not in stopwords]
    if len(content_tokens) == 0:
        return False

    # --- 6. Reject candidates that are mostly stopwords ---
    # e.g. "van die", "ya le", etc.
    stopword_ratio = 1 - (len(content_tokens) / len(tokens))
    if stopword_ratio > 0.6:
        return False

    # --- 7. Reject bad edges (common noise pattern) ---
    # leading/trailing stopwords are very often fragments
    if tokens[0] in stopwords or tokens[-1] in stopwords:
        # allow if there are strong content tokens inside (e.g. "van binnelandse sake")
        if len(content_tokens) < 2:
            return False

    # --- 8. Reject repeated-token junk ---
    # e.g. "ya ya", "van van"
    if len(set(tokens)) == 1 and len(tokens) > 1:
        return False

    # --- 9. Limit extremely long phrases (safety) ---
    if len(tokens) > 6:
        return False

    return True


def induce_lexicon_from_pairs(
    pairs_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    config,
    source_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Induce candidate target-language lexicon entries from English-aligned pairs.
    """

    induced_cols = [
        "surface",
        "lang",
        "kind",
        "semantic_type",
        "local_marker",
        "provenance",
        "score",
        "count",
        "seed_surface",
        "seed_kind",
        "seed_semantic_type",
        "induced_from_source",
    ]
    stats_cols = [
        "source",
        "seed_surface",
        "seed_kind",
        "seed_semantic_type",
        "language",
        "seed_sentence_count",
        "candidate_count",
    ]

    source_df = pairs_df[pairs_df["source"] == source_name].copy()
    if source_df.empty:
        return pd.DataFrame(columns=induced_cols), pd.DataFrame(columns=stats_cols)

    english_seed_df = seed_df[seed_df["lang"] == "eng"].copy()
    if english_seed_df.empty:
        raise ValueError("Seed lexicon must include English seed entries.")

    stopword_cache = {
        lang: _load_stopwords(path) for lang, path in config.stopword_files.items()
    }

    source_thresholds = config.source_thresholds.get(source_name)
    if source_thresholds is None:
        min_candidate_count = 2
        min_candidate_score = 0.1
        top_k_per_seed_lang = 50
    else:
        min_candidate_count = source_thresholds.min_candidate_count
        min_candidate_score = source_thresholds.min_candidate_score
        top_k_per_seed_lang = source_thresholds.top_k_per_seed_lang

    background_by_lang: dict[str, Counter[str]] = {}
    for lang, group in source_df.groupby("language"):
        background_by_lang[lang] = count_ngrams(
            group["other_sentence"].fillna("").tolist(),
            max_n=config.max_ngram,
        )

    induced_rows: list[dict[str, object]] = []
    stats_rows: list[dict[str, object]] = []

    for _, seed in english_seed_df.iterrows():
        seed_surface = str(seed["surface"])
        seed_kind = str(seed["kind"])
        semantic_type = str(seed["semantic_type"])

        seed_hits = source_df[
            source_df["english"].fillna("").str.casefold().str.contains(seed_surface.casefold(), regex=False)
        ]
        if seed_hits.empty:
            continue

        for lang, lang_group in seed_hits.groupby("language"):
            stopwords = stopword_cache.get(lang, set())
            candidate_counter: Counter[str] = Counter()

            for sent in lang_group["other_sentence"].fillna("").tolist():
                sent_counts = count_ngrams([sent], max_n=config.max_ngram)
                for ng in sent_counts.keys():
                    if not _valid_candidate(ng, config.min_target_token_chars, stopwords):
                        continue
                    if config.prune.enabled and not _apply_candidate_shape_filters(
                        ng, seed_kind, stopwords, config.prune
                    ):
                        continue
                    candidate_counter[ng] += 1

            scored_candidates = score_candidates(
                seed_surface=seed_surface,
                seed_sentence_count=len(lang_group),
                candidate_counts=candidate_counter,
                background_counts=background_by_lang[lang],
            )

            kept = 0
            for candidate, score in scored_candidates[:top_k_per_seed_lang]:
                count = candidate_counter[candidate]
                if count < min_candidate_count or score < min_candidate_score:
                    continue

                induced_rows.append(
                    {
                        "surface": candidate,
                        "lang": lang,
                        "kind": seed_kind,
                        "semantic_type": semantic_type,
                        "local_marker": bool(seed["local_marker"]),
                        "provenance": f"induced::{source_name}",
                        "score": float(score),
                        "count": int(count),
                        "seed_surface": seed_surface,
                        "seed_kind": seed_kind,
                        "seed_semantic_type": semantic_type,
                        "induced_from_source": source_name,
                    }
                )
                kept += 1

            stats_rows.append(
                {
                    "source": source_name,
                    "seed_surface": seed_surface,
                    "seed_kind": seed_kind,
                    "seed_semantic_type": semantic_type,
                    "language": lang,
                    "seed_sentence_count": len(lang_group),
                    "candidate_count": kept,
                }
            )

    induced_df = pd.DataFrame(induced_rows, columns=induced_cols)
    stats_df = pd.DataFrame(stats_rows, columns=stats_cols)

    if config.prune.enabled and not induced_df.empty and config.prune.suppress_subphrases:
        induced_df = prune_subphrases(
            induced_df,
            subphrase_score_ratio=config.prune.subphrase_score_ratio,
        )

    return induced_df, stats_df


def combine_lexicons(seed_df: pd.DataFrame, induced_dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [seed_df] + [df for df in induced_dfs if not df.empty]
    combined = pd.concat(frames, ignore_index=True)
    combined["surface"] = combined["surface"].str.casefold()
    # Keep the strongest row per surface/language/kind/source provenance combo.
    combined = (
        combined.sort_values(["surface", "lang", "kind", "score", "count"], ascending=[True, True, True, False, False])
        .drop_duplicates(subset=["surface", "lang", "kind", "semantic_type", "provenance"], keep="first")
        .reset_index(drop=True)
    )
    return combined


from pathlib import Path
import pandas as pd


REQUIRED_COMPARE_COLS = ["surface", "lang", "kind", "semantic_type"]


def _ensure_compare_schema(df: pd.DataFrame, score_prefix: str) -> pd.DataFrame:
    """
    Ensure the lexicon dataframe has the columns needed for source comparison.

    This prevents merge failures when one source yields no candidates or returns
    a slightly different schema.
    """
    if df is None or df.empty:
        out = pd.DataFrame(columns=REQUIRED_COMPARE_COLS + [f"{score_prefix}_score", f"{score_prefix}_count"])
        return out

    df = df.copy()

    # Normalise common alternate column names.
    rename_map = {}
    if "language" in df.columns and "lang" not in df.columns:
        rename_map["language"] = "lang"
    if "type" in df.columns and "semantic_type" not in df.columns:
        rename_map["type"] = "semantic_type"
    df = df.rename(columns=rename_map)

    # Add missing required columns as empty strings so merge can proceed.
    for col in REQUIRED_COMPARE_COLS:
        if col not in df.columns:
            df[col] = ""

    # Ensure optional score/count columns exist.
    if "score" not in df.columns:
        df["score"] = 0.0
    if "count" not in df.columns:
        df["count"] = 0

    return df


def compare_lexicon_sources(
    aut_df: pd.DataFrame,
    vuk_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    aut_df = _ensure_compare_schema(aut_df, "aut")
    vuk_df = _ensure_compare_schema(vuk_df, "vuk")

    aut_keyed = aut_df.copy()
    aut_keyed["aut_present"] = True
    aut_keyed = aut_keyed.rename(columns={"score": "aut_score", "count": "aut_count"})

    vuk_keyed = vuk_df.copy()
    vuk_keyed["vuk_present"] = True
    vuk_keyed = vuk_keyed.rename(columns={"score": "vuk_score", "count": "vuk_count"})

    merge_cols = ["surface", "lang", "kind", "semantic_type"]

    keep_cols_aut = merge_cols + ["aut_present", "aut_score", "aut_count"]
    keep_cols_vuk = merge_cols + ["vuk_present", "vuk_score", "vuk_count"]

    comparison = aut_keyed[keep_cols_aut].merge(
        vuk_keyed[keep_cols_vuk],
        how="outer",
        on=merge_cols,
    )

    comparison["aut_present"] = comparison["aut_present"].fillna(False)
    comparison["vuk_present"] = comparison["vuk_present"].fillna(False)
    comparison["aut_score"] = comparison["aut_score"].fillna(0.0)
    comparison["vuk_score"] = comparison["vuk_score"].fillna(0.0)
    comparison["aut_count"] = comparison["aut_count"].fillna(0).astype(int)
    comparison["vuk_count"] = comparison["vuk_count"].fillna(0).astype(int)
    comparison["present_in_both"] = comparison["aut_present"] & comparison["vuk_present"]

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_path, index=False)

    return comparison

def top_candidates_by_language_and_kind(
    df: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    sort_df = df.sort_values(
        by=["lang", "kind", "score", "count", "surface"],
        ascending=[True, True, False, False, True],
    ).copy()

    top_df = (
        sort_df.groupby(["lang", "kind"], group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return top_df