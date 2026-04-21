import pandas as pd
import numpy as np

def sample_gold200(examples: pd.DataFrame, family_col="family_id", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Work at family level; each family has 4 rows
    fam = (examples.groupby(family_col).agg(language=("language","first"),
                label=("label","mean"), 
                is_local_task=("is_local_task","max"),
                n=("id","count")).reset_index())
    # Filter to complete families
    fam = fam[fam["n"] == 4].copy()
    # Stratify by language; allocate families proportional but ensure at least 1 per language present
    langs = fam["language"].value_counts()
    # choose 50 families => 200 rows
    target_families = 50
    selected_fams = []
    for lang, cnt in langs.items():
        k = max(1, int(round(target_families * (cnt / len(fam)))))
        pool = fam[fam["language"] == lang][family_col].tolist()
        if len(pool) <= k:
            chosen = pool
        else:
            chosen = rng.choice(pool, size=k, replace=False).tolist()
        selected_fams.extend(chosen)

    # Trim/expand to exactly target_families
    selected_fams = list(dict.fromkeys(selected_fams)) # dedup keep order
    if len(selected_fams) > target_families:
        selected_fams = selected_fams[:target_families]
    elif len(selected_fams) < target_families:
        remaining = [f for f in fam[family_col].tolist() if f not in set(selected_fams)]
        add = rng.choice(remaining, size=(target_families-len(selected_fams)), replace=False).tolist()
        selected_fams.extend(add)
    gold = examples[examples[family_col].isin(selected_fams)].copy()
    return gold