# src/relation_check.py
from curses import window
from typing import Set

def relation_window(tokens: list[str], inst_terms: Set[str], act_terms: Set[str], window: int=8) -> bool:
    inst_pos = [i for i,t in enumerate(tokens) if t in inst_terms]
    act_pos = [i for i,t in enumerate(tokens) if t in act_terms]
    for i in inst_pos:
        for j in act_pos:
            if abs(i-j) <= window:
                return True
        return False

def relation_spacy_en(doc, inst_surfaces: set[str], act_lemmas: set[str]) -> bool:
    inst_tokens = [t for t in doc if t.text.casefold() in inst_surfaces]
    act_tokens = [t for t in doc if t.pos_ == "VERB" and t.lemma_.casefold() in act_lemmas]
    if not inst_tokens or not act_tokens:
        return False
    for inst in inst_tokens:
        if inst.dep_ in {"nsubj","nsubjpass","obj","iobj","pobj"}:
            if inst.head in act_tokens or inst.head.head in act_tokens:
                return True
    return False