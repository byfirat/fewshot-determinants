import pandas as pd


def select_domain_pair(df: pd.DataFrame, source_domain: str, target_domain: str):
    source_df = df[df["domain"] == source_domain].copy()
    target_df = df[df["domain"] == target_domain].copy()
    if source_df.empty or target_df.empty:
        raise ValueError("Source or target domain has no rows.")
    return source_df, target_df
