# src/segment.py
import re
import spacy

_SENT_SPLIT_FALLBACK = re.compile(r"(?<=[.!?])\s+")

def make_spacy_en():
    return spacy.load("en_core_web_sm", exclude=["ner"])

def make_sentencizer():
    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    return nlp

def segment_text(text_norm: str, lang_iso3: str, nlp_en=None, nlp_sent=None) ->list[str]:
    if lang_iso3 == "eng":
        doc = nlp_en(text_norm)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    # fallback: sentencizer + regex fallback
    doc = nlp_sent(text_norm)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if len(sents) <= 1:
        sents = [s.strip() for s in _SENT_SPLIT_FALLBACK.split(text_norm) if s.strip()]
    return sents