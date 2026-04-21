from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from rapidfuzz.fuzz import ratio as rf_ratio
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass(frozen=True)
class SwapCandidate:
    pert_type: str
    text: str
    rf: float
    cos: float
    edits: int
    ok: bool

def token_edit_count(a: str, b: str) -> int: 
    ta, tb = a.split(), b.split()
    # cheap proxy: count positions with different token, plus length delta
    n = min(len(ta), len(tb))
    diffs = sum(1 for i in range(n) if ta[i] != tb[i])
    diffs += abs(len(ta) - len(tb))
    return diffs

def cosine_sim(embed_a: np.ndarray, embed_b: np.ndarray) -> float:
    # embeddings assumed normalized
    return float(np.dot(embed_a, embed_b))

def generate_institution_swap(base: str, inst_surface: str, repl_surface: str) -> str:
    # minimal, surface-level replacement
    return base.replace(inst_surface, repl_surface)

def score_candidate(base: str, cand: str, model: SentenceTransformer, rf_thr=88.0, cos_thr=0.82, edit_thr=4) -> Tuple[float,float,int,bool]:
    rf = rf_ratio(base, cand)
    emb = model.encode([base, cand], normalize_embeddings=True)
    cos = cosine_sim(emb[0], emb[1])
    edits = token_edit_count(base, cand)
    ok = (rf >= rf_thr) and (cos >= cos_thr) and (edits <= edit_thr)
    return rf, cos, edits, ok