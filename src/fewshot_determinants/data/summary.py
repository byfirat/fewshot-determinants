from __future__ import annotations

import pandas as pd


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["domain", "split", "label"]) 
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["domain", "split", "label"])
        .reset_index(drop=True)
    )
