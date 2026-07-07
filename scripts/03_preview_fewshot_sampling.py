import pandas as pd

from fewshot_determinants.data.sampler import sample_fewshot_by_class
from fewshot_determinants.data.splitters import select_domain_pair
from fewshot_determinants.paths import INTERIM_DIR


def main():
    df = pd.read_csv(INTERIM_DIR / "toy_multidomain_sentiment.csv")
    train_df = df[df["split"] == "train"].copy()
    _, target_df = select_domain_pair(train_df, source_domain="books", target_domain="electronics")
    sampled = sample_fewshot_by_class(target_df, shots_per_class=4, seed=42)
    print("Preview sampled few-shot set for books -> electronics")
    print(sampled.head(8).to_string(index=False))
    print("Counts by label:")
    print(sampled.groupby("label").size())


if __name__ == "__main__":
    main()
