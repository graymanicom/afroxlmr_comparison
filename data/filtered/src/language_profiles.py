from dataclasses import dataclass
from typing import Literal, Optional

RelationMode = Literal["spacy_dep", "ud_dep", "window_frames"]
SegmentMode = Literal["spacy_sents", "stanza_sents", "sentencizer_regex"]

@dataclass(frozen=True)
class LanguageProfile:
    iso3: str
    name: str
    segment_mode: SegmentMode
    relation_mode: RelationMode
    ws_min: int
    ws_max: int
    sp_min: int
    sp_max: int

PROFILES = {
"eng": LanguageProfile("eng","English","spacy_sents","spacy_dep", 8,30, 12, 64),
"afr": LanguageProfile("afr","Afrikaans","stanza_sents","ud_dep", 8,30, 12, 64),
# Bantu fallback defaults
"zul": LanguageProfile("zul","isiZulu","sentencizer_regex","window_frames", 6,28,12,64),
"xho": LanguageProfile("xho","isiXhosa","sentencizer_regex","window_frames", 6,28, 12,64),
"nbl": LanguageProfile("nbl","isiNdebele","sentencizer_regex","window_frames", 6,28,12,64),
"ssw": LanguageProfile("ssw","siSwati","sentencizer_regex","window_frames",6,28, 12,64),
"nso": LanguageProfile("nso","Sepedi","sentencizer_regex","window_frames",6,28, 12,64),
"sot": LanguageProfile("sot","Sesotho","sentencizer_regex","window_frames",6,28, 12,64),
"tsn": LanguageProfile("tsn","Setswana","sentencizer_regex","window_frames", 6,28, 12,64),
"ven": LanguageProfile("ven","Tshivenda","sentencizer_regex","window_frames", 6,28, 12,64),
"tso": LanguageProfile("tso","Xitsonga","sentencizer_regex","window_frames", 6,28, 12,64),
}