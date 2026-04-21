# src/filter_candidates.py
from pydoc import text
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.\-]+@[\w.\-]+\.\w+\b")
BULLET_RE = re.compile(r"^\s*([\-\*\u2022]|\d+[\.\)])\s+")

def is_metadata_like(text: str) -> bool:
    if URL_RE.search(text) or EMAIL_RE.search(text):
        return True
    if BULLET_RE.search(text):
        return True
    
    digits = sum(ch.isdigit() for ch in text)
    uppers = sum(ch.isupper() for ch in text)
    n = max(len(text), 1)
    digit_ratio = digits / n
    upper_ratio = uppers / n
    if digit_ratio > 0.25:
        return True
    if upper_ratio > 0.85 and len(text.split()) < 12:
        return True
    return False

ANAPHOR_START = {
"eng": re.compile(r"^(this|that|these|those|it|they|he|she|we)\b", re.I),
"afr": re.compile(r"^(dit|daardie|hierdie|hulle|hy|sy|ons)\b", re.I),
# Extend per language after pilot
}

def self_containment_score(text: str, lang: str, arg_terms: set[str], access_terms: set[str]) -> int:
    score = 0
    words = text.split()
    if words:
        rx = ANAPHOR_START.get(lang)
        if (rx is None) or (not rx.match(words[0])):
            score += 1
        low = text.casefold()
        if any(t in low for t in arg_terms):
            score += 1
        if any(t in low for t in access_terms):
            score += 1
        return min(score, 3)