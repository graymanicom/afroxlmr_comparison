# src/dedup_embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer

def embedding_dedup(texts: list[str], ids: list[str], model_name: str, cos_thr: float=0.95) -> list[str]:
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    keep = []
    dropped = set()
    for i, id_i in enumerate(ids):
        if id_i in dropped:
            continue
    keep.append(id_i)
    sims = emb @ emb[i]
    for j, s in enumerate(sims):
        if j != i and s >= cos_thr:
            dropped.add(ids[j])
    return keep