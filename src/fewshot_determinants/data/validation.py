import pandas as pd

REQUIRED_COLUMNS = {"text", "label", "domain", "split"}


def validate_dataframe(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("DataFrame is empty.")
