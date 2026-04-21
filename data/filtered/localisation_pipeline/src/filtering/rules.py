from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import EnglishFilteringConfig, PairedLanguageFilteringConfig
from src.utils.text import (
    has_single_sentence_shape,
    is_metadata_like,
    is_temporally_brittle,
    starts_with_bare_english_anaphor,
    tokenize,
    whitespace_token_count,
)


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    score: int
    reasons: list[str]


def build_lexicon_lookup(lexicon_df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Turn the lexicon table into a lookup by language and semantic kind."""
    lookup: dict[str, dict[str, list[str]]] = {}
    for lang, group in lexicon_df.groupby("lang"):
        lookup[lang] = {}
        for kind, kind_group in group.groupby("kind"):
            surfaces = sorted({str(surface) for surface in kind_group["surface"].dropna().tolist()})
            lookup[lang][kind] = surfaces
    return lookup


def lexicon_hits(text: str, surfaces: list[str]) -> list[str]:
    low = text.casefold()
    return [surface for surface in surfaces if surface in low]


def relation_window(tokens: list[str], inst_terms: set[str], act_terms: set[str], window: int = 8) -> bool:
    inst_pos = [i for i, t in enumerate(tokens) if t in inst_terms]
    act_pos = [i for i, t in enumerate(tokens) if t in act_terms]
    for i in inst_pos:
        for j in act_pos:
            if abs(i - j) <= window:
                return True
    return False


def score_english_candidate(
    english_text: str,
    source: str,
    english_lookup: dict[str, list[str]],
    config: EnglishFilteringConfig,
) -> ValidationResult:
    reasons: list[str] = []
    score = 0

    if config.allow_sources and source not in config.allow_sources:
        reasons.append("source_not_allowed")
        return ValidationResult(False, score, reasons)

    wc = whitespace_token_count(english_text)
    if wc < config.min_words or wc > config.max_words:
        reasons.append("word_count_out_of_range")
    else:
        score += 1

    if len(english_text) < config.min_chars or len(english_text) > config.max_chars:
        reasons.append("char_count_out_of_range")
    else:
        score += 1

    if is_metadata_like(
        english_text,
        max_digit_ratio=config.max_digit_ratio,
        max_upper_ratio=config.max_upper_ratio_short,
    ):
        reasons.append("metadata_like")
    else:
        score += 1

    if has_single_sentence_shape(english_text):
        score += 1
    else:
        reasons.append("not_single_sentence_like")

    if starts_with_bare_english_anaphor(english_text):
        reasons.append("starts_with_bare_anaphor")
    else:
        score += 1

    if is_temporally_brittle(english_text):
        reasons.append("temporally_brittle")
    else:
        score += 1

    inst_hits = lexicon_hits(english_text, english_lookup.get("institution", []))
    action_hits = lexicon_hits(english_text, english_lookup.get("action", []))
    if config.require_institution_hit and not inst_hits:
        reasons.append("no_institution_hit")
    elif inst_hits:
        score += 1

    if config.require_action_hit and not action_hits:
        reasons.append("no_action_hit")
    elif action_hits:
        score += 1

    passed = score >= 5 and not {"metadata_like", "temporally_brittle", "not_single_sentence_like"}.intersection(reasons)
    return ValidationResult(passed, score, reasons)


def validate_paired_language(
    other_text: str,
    language: str,
    lexicon_lookup: dict[str, dict[str, list[str]]],
    config: PairedLanguageFilteringConfig,
) -> ValidationResult:
    reasons: list[str] = []
    score = 0

    if config.require_nonempty and not other_text.strip():
        reasons.append("empty_other_sentence")
        return ValidationResult(False, score, reasons)

    wc = whitespace_token_count(other_text)
    if wc < config.min_words or wc > config.max_words:
        reasons.append("other_word_count_out_of_range")
    else:
        score += 1

    lang_lookup = lexicon_lookup.get(language, {})
    inst_hits = lexicon_hits(other_text, lang_lookup.get("institution", []))
    action_hits = lexicon_hits(other_text, lang_lookup.get("action", []))
    locality_hits = lexicon_hits(other_text, lang_lookup.get("locality", []))

    if inst_hits:
        score += 1
    if action_hits:
        score += 1
    if locality_hits:
        score += 1

    if inst_hits and action_hits:
        token_list = tokenize(other_text)
        if relation_window(token_list, set(tokenize(" ".join(inst_hits))), set(tokenize(" ".join(action_hits))), window=config.relation_window):
            score += 1
        else:
            reasons.append("no_relation_window_match")
    elif not (inst_hits or locality_hits):
        reasons.append("no_non_english_grounding_hit")

    passed = score >= config.min_validation_score
    return ValidationResult(passed, score, reasons)
