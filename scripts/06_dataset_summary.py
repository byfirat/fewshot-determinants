import pandas as pd

from fewshot_determinants.data.summary import dataset_summary
from fewshot_determinants.paths import CONFIGS_DIR, PROCESSED_DIR
from fewshot_determinants.utils.io import load_yaml


def main():
    cfg = load_yaml(CONFIGS_DIR / "data.yaml")
    filename = cfg["preparation"]["output_filename"]
    path = PROCESSED_DIR / filename
    df = pd.read_csv(path)
    print(dataset_summary(df).to_string(index=False))
    print("\nUnique domains:", sorted(df["domain"].unique().tolist()))
    print("Total rows:", len(df))


if __name__ == "__main__":
    main()
