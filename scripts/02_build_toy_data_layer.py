import pandas as pd

from fewshot_determinants.data.validation import validate_dataframe
from fewshot_determinants.paths import INTERIM_DIR
from fewshot_determinants.utils.io import ensure_dir


def main():
    rows = []
    domains = ["books", "dvd", "electronics", "kitchen"]
    for domain in domains:
        for i in range(20):
            rows.append({"text": f"Positive sample {i} from {domain}", "label": 1, "domain": domain, "split": "train"})
            rows.append({"text": f"Negative sample {i} from {domain}", "label": 0, "domain": domain, "split": "train"})
        for i in range(10):
            rows.append({"text": f"Positive test {i} from {domain}", "label": 1, "domain": domain, "split": "test"})
            rows.append({"text": f"Negative test {i} from {domain}", "label": 0, "domain": domain, "split": "test"})

    df = pd.DataFrame(rows)
    validate_dataframe(df)
    ensure_dir(INTERIM_DIR)
    out_path = INTERIM_DIR / "toy_multidomain_sentiment.csv"
    df.to_csv(out_path, index=False)
    print(f"Toy data saved to: {out_path}")
    print(df.groupby(["domain", "split", "label"]).size())


if __name__ == "__main__":
    main()
