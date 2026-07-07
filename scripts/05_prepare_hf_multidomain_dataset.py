from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from fewshot_determinants.data.hf_source import (
    combine_text_columns,
    hf_to_dataframe,
    load_hf_split,
)
from fewshot_determinants.paths import CONFIGS_DIR, PROCESSED_DIR
from fewshot_determinants.utils.io import ensure_dir, load_yaml


def split_one_domain(
    df: pd.DataFrame,
    domain_name: str,
    max_train: int,
    max_validation: int,
    max_test: int,
    seed: int = 42,
) -> pd.DataFrame:
    domain_df = df[df["domain"] == domain_name].copy()

    if domain_df.empty:
        raise ValueError(f"Domain is empty: {domain_name}")

    domain_df = domain_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    total_needed = max_train + max_validation + max_test
    if len(domain_df) > total_needed:
        domain_df = domain_df.iloc[:total_needed].copy()

    # önce test ayır
    test_size = min(max_test, len(domain_df))
    if test_size == 0:
        raise ValueError(f"No rows available for test split in domain: {domain_name}")

    rest_df, test_df = train_test_split(
        domain_df,
        test_size=test_size,
        random_state=seed,
        stratify=domain_df["label"] if domain_df["label"].nunique() > 1 else None,
    )

    # sonra validation ayır
    val_size = min(max_validation, len(rest_df))
    if val_size == 0:
        raise ValueError(f"No rows available for validation split in domain: {domain_name}")

    train_df, val_df = train_test_split(
        rest_df,
        test_size=val_size,
        random_state=seed,
        stratify=rest_df["label"] if rest_df["label"].nunique() > 1 else None,
    )

    # train boyutunu sınırla
    if len(train_df) > max_train:
        train_df = train_df.sample(
            n=max_train,
            random_state=seed,
            stratify=train_df["label"] if train_df["label"].nunique() > 1 else None,
        )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def main():
    cfg = load_yaml(CONFIGS_DIR / "data.yaml")
    source_cfg = cfg["source"]
    prep_cfg = cfg["preparation"]

    hf_ds = load_hf_split(
        dataset_name=source_cfg["dataset_name"],
        dataset_config=source_cfg["dataset_config"],
        split=source_cfg["inspect_split"],
    )

    raw_df = hf_to_dataframe(hf_ds).copy()

    # temel temizlik
    raw_df["product_type"] = raw_df["product_type"].astype(str).str.strip()
    raw_df["rating"] = pd.to_numeric(raw_df["rating"], errors="coerce")
    raw_df["title"] = raw_df["title"].fillna("").astype(str)
    raw_df["review_text"] = raw_df["review_text"].fillna("").astype(str)

    raw_df = raw_df.dropna(subset=["product_type", "rating"]).copy()

    selected_domains = prep_cfg["selected_domains"]
    raw_df = raw_df[raw_df["product_type"].isin(selected_domains)].copy()

    negative_stars = set(prep_cfg["negative_stars"])
    positive_stars = set(prep_cfg["positive_stars"])

    raw_df["rating_int"] = raw_df["rating"].round().astype(int)

    if prep_cfg["drop_neutral"]:
        raw_df = raw_df[
            raw_df["rating_int"].isin(negative_stars.union(positive_stars))
        ].copy()

    raw_df["label"] = raw_df["rating_int"].apply(
        lambda x: 0 if x in negative_stars else 1
    )

    raw_df["text"] = combine_text_columns(raw_df, prep_cfg["text_fields"])
    raw_df["text"] = raw_df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    raw_df = raw_df[raw_df["text"] != ""].copy()

    standardized = raw_df[["text", "label", "product_type"]].rename(
        columns={"product_type": "domain"}
    )

    all_parts = []
    for domain in selected_domains:
        domain_part = split_one_domain(
            standardized,
            domain_name=domain,
            max_train=prep_cfg["max_train_per_domain"],
            max_validation=prep_cfg["max_validation_per_domain"],
            max_test=prep_cfg["max_test_per_domain"],
            seed=42,
        )
        all_parts.append(domain_part)

    final_df = pd.concat(all_parts, ignore_index=True)

    ensure_dir(PROCESSED_DIR)
    out_path = PROCESSED_DIR / prep_cfg["output_filename"]
    final_df.to_csv(out_path, index=False)

    print(f"Saved processed dataset to: {out_path}")
    print("\nCounts by domain / split / label:")
    print(final_df.groupby(["domain", "split", "label"]).size())


if __name__ == "__main__":
    main()