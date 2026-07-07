from __future__ import annotations

from typing import Iterable

import pandas as pd

from fewshot_determinants.data.validation import validate_dataframe


def map_rating_to_label(
    stars: pd.Series,
    negative_stars: Iterable[int],
    positive_stars: Iterable[int],
    drop_neutral: bool = True,
) -> pd.Series:
    neg = set(negative_stars)
    pos = set(positive_stars)

    def mapper(value: int):
        if value in neg:
            return 0
        if value in pos:
            return 1
        return None if drop_neutral else -1

    return stars.map(mapper)


def standardize_reviews_dataframe(
    df: pd.DataFrame,
    *,
    text_series: pd.Series,
    label_series: pd.Series,
    domain_column: str,
    split_name: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "text": text_series.astype(str).str.strip(),
            "label": label_series,
            "domain": df[domain_column].astype(str),
            "split": split_name,
        }
    )
    out = out.dropna(subset=["label"])
    out = out[out["text"].str.len() > 0].copy()
    out["label"] = out["label"].astype(int)
    validate_dataframe(out)
    return out.reset_index(drop=True)


def limit_per_domain(df: pd.DataFrame, domain: str, max_rows: int, seed: int) -> pd.DataFrame:
    part = df[df["domain"] == domain].copy()
    if max_rows is None or len(part) <= max_rows:
        return part
    return part.sample(n=max_rows, random_state=seed).reset_index(drop=True)
