# src/dedup.py
import hashlib
from datasketch import MinHash, MinHashLSH

def exact_hash(text_norm: str) -> str:
    return hashlib.sha1(text_norm.encode("utf-8")).hexdigest()

def build_minhash(tokens: set[str], num_perm: int=128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for t in tokens:
        mh.update(t.encode("utf-8"))
    return mh

def lsh_dedup(records: list[dict], text_key: str="text_norm", threshold: float=0.9) -> list[dict]:
    # records must have stable 'id' field
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    sigs = {}

    for r in records:
        toks = set(r[text_key].casefold().split())
        mh = build_minhash(toks)
        sigs[r["id"]] = mh
        lsh.insert(r["id"], mh)

    kept = []
    seen = set()
    for r in records:
        if r["id"] in seen:
            continue
    nbrs = lsh.query(sigs[r["id"]])
    kept.append(r)
    for n in nbrs:
        seen.add(n)
    return kept
# keep the first, mark others as duplicates