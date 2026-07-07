from __future__ import annotations

from typing import Iterable
import re

import pandas as pd
from datasets import load_dataset


def load_hf_split(dataset_name: str, dataset_config: str, split: str):
    return load_dataset(dataset_name, dataset_config, split=split)


def _to_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    return str(value)


def _extract_tag(block: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, block, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_review_blob(blob: str) -> list[dict]:
    review_blocks = re.findall(r"<review>\s*(.*?)\s*</review>", blob, flags=re.DOTALL)
    rows = []

    for block in review_blocks:
        row = {
            "unique_id": _extract_tag(block, "unique_id"),
            "asin": _extract_tag(block, "asin"),
            "product_name": _extract_tag(block, "product_name"),
            "product_type": _extract_tag(block, "product_type"),
            "helpful": _extract_tag(block, "helpful"),
            "rating": _extract_tag(block, "rating"),
            "title": _extract_tag(block, "title"),
            "date": _extract_tag(block, "date"),
            "reviewer": _extract_tag(block, "reviewer"),
            "reviewer_location": _extract_tag(block, "reviewer_location"),
            "review_text": _extract_tag(block, "review_text"),
        }

        # Boş satırları alma
        if row["product_type"] and row["review_text"]:
            rows.append(row)

    return rows


def hf_to_dataframe(hf_dataset) -> pd.DataFrame:
    parsed_rows = []

    for item in hf_dataset:
        blob = _to_text(item.get("review", ""))

        if not blob.strip():
            continue

        rows = _parse_review_blob(blob)
        for row in rows:
            row["source_key"] = item.get("__key__", "")
            row["source_url"] = item.get("__url__", "")
            parsed_rows.append(row)

    return pd.DataFrame(parsed_rows)


def available_domains(df: pd.DataFrame, column: str = "product_type") -> pd.Series:
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. Available columns: {list(df.columns)}"
        )
    return df[column].value_counts().sort_values(ascending=False)


def combine_text_columns(df: pd.DataFrame, text_fields: Iterable[str]) -> pd.Series:
    text_fields = list(text_fields)
    missing = [field for field in text_fields if field not in df.columns]
    if missing:
        raise ValueError(
            f"Missing text field(s): {missing}. Available columns: {list(df.columns)}"
        )

    combined = df[text_fields[0]].fillna("").astype(str).str.strip()
    for field in text_fields[1:]:
        combined = combined + " " + df[field].fillna("").astype(str).str.strip()

    return combined.str.replace(r"\s+", " ", regex=True).str.strip()