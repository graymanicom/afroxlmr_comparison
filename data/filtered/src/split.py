import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def assign_splits(df: pd.DataFrame, group_col: str, strata_col: str, seed: int=42) -> pd.DataFrame:
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    groups = df[group_col].to_numpy()
    strata = df[strata_col].to_numpy()

    fold_id = np.full(len(df), -1, dtype=int)
    for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(df)), strata, groups=groups)):
        fold_id[test_idx] = k
        
    df = df.copy()
    df["fold"] = fold_id
    # 70/15/15 split via folds: first 14 folds train, next 3 dev, last 3 test
    df["split"] = "train"
    df.loc[df["fold"].isin([14,15,16]), "split"] = "dev"
    df.loc[df["fold"].isin([17,18,19]), "split"] = "test"
    return df.drop(columns=["fold"])