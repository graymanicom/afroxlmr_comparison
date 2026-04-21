from typing import Dict, List

def lexicon_match(text_norm: str, lex: Dict[str, str]) -> List[str]:
    low = text_norm.casefold()
    hits = []
    for surface, tag in lex.items():
        if surface in low:
            hits.append(tag)
    return hits