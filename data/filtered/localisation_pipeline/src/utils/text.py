from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from unidecode import unidecode

import regex

TOKEN_RE = regex.compile(r"[\p{L}\p{N}]+(?:[-'][\p{L}\p{N}]+)?", regex.UNICODE)
MULTISPACE_RE = regex.compile(r"\s+")
URL_RE = regex.compile(r"https?://\S+|www\.\S+", regex.I)
EMAIL_RE = regex.compile(r"\b[\w.\-]+@[\w.\-]+\.\w+\b")
BULLET_RE = regex.compile(r"^\s*([\-*•]|\d+[\.)])\s+")
ANAPHOR_START_ENG_RE = regex.compile(r"^(this|that|these|those|it|they|he|she|we)\b", regex.I)
QUESTION_OR_EXCL_RE = regex.compile(r"[!?]")
DATE_LIKE_RE = regex.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|january|february|march|april|may|june|"
    r"july|august|september|october|november|december|currently|this year|deadline|close on)\b",
    regex.I,
)


def normalize_text(text: str) -> str:
    """Light normalization suitable for aligned sentence pairs.

    This is intentionally conservative: unlike the PDF-specific pipeline in the report,
    we avoid aggressive cleanup that could damage aligned text.
    """
    text = text.replace("\u00ad", "")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def ascii_fold(text: str) -> str:
    return unidecode(text).casefold()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def whitespace_token_count(text: str) -> int:
    return len(text.split())


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isdigit() for ch in text) / len(text)


def upper_ratio(text: str) -> float:
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return 0.0
    return sum(ch.isupper() for ch in alpha) / len(alpha)


def is_metadata_like(text: str, max_digit_ratio: float = 0.25, max_upper_ratio: float = 0.85) -> bool:
    if URL_RE.search(text) or EMAIL_RE.search(text) or BULLET_RE.search(text):
        return True
    if digit_ratio(text) > max_digit_ratio:
        return True
    if upper_ratio(text) > max_upper_ratio and whitespace_token_count(text) < 12:
        return True
    return False


def has_single_sentence_shape(text: str) -> bool:
    """A conservative single-sentence heuristic.

    We reject explicit questions/exclamations and strings with multiple terminal punctuation chunks.
    """
    if QUESTION_OR_EXCL_RE.search(text):
        return False
    terminal_count = sum(text.count(mark) for mark in ".;:")
    return terminal_count <= 2


def starts_with_bare_english_anaphor(text: str) -> bool:
    return bool(ANAPHOR_START_ENG_RE.match(text.strip()))


def is_temporally_brittle(text: str) -> bool:
    return bool(DATE_LIKE_RE.search(text))


def generate_ngrams(tokens: list[str], max_n: int = 4) -> list[str]:
    grams: list[str] = []
    for n in range(1, max_n + 1):
        for i in range(0, len(tokens) - n + 1):
            grams.append(" ".join(tokens[i : i + n]))
    return grams


def score_candidates(
    seed_surface: str,
    seed_sentence_count: int,
    candidate_counts: Counter[str],
    background_counts: Counter[str],
) -> list[tuple[str, float]]:
    """Return PMI-like scores for candidate target terms.

    The exact measure is intentionally simple and interpretable. It rewards
    repeated co-occurrence with a seed while downweighting globally frequent n-grams.
    """
    scored: list[tuple[str, float]] = []
    for candidate, count in candidate_counts.items():
        bg = background_counts[candidate]
        # Smoothed PMI-ish scoring that behaves well on moderate corpus sizes.
        score = (count / max(seed_sentence_count, 1)) * math.log(1 + (count / max(bg, 1e-6)))
        scored.append((candidate, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def count_ngrams(sentences: Iterable[str], max_n: int = 4) -> Counter[str]:
    counter: Counter[str] = Counter()
    for sentence in sentences:
        counter.update(generate_ngrams(tokenize(sentence), max_n=max_n))
    return counter
