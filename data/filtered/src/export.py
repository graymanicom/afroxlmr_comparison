# src/export.py
import pandas as pd

REQUIRED = ["id","text","label","language","is_local_task","pair_id","pair_role","split"]

def export_dataset(df: pd.DataFrame, out_csv: str) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required cols: {missing}")
    df[REQUIRED].to_csv(out_csv, index=False)