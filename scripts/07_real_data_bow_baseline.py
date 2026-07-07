import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fewshot_determinants.data.sampler import sample_fewshot_by_class
from fewshot_determinants.evaluation.metrics import compute_classification_metrics
from fewshot_determinants.paths import CONFIGS_DIR, PROCESSED_DIR
from fewshot_determinants.utils.io import load_yaml


def main():
    cfg = load_yaml(CONFIGS_DIR / "data.yaml")
    filename = cfg["preparation"]["output_filename"]
    df = pd.read_csv(PROCESSED_DIR / filename)

    source_domain = cfg["preparation"]["selected_domains"][0]
    target_domain = cfg["preparation"]["selected_domains"][1]

    source_train = df[(df["domain"] == source_domain) & (df["split"] == "train")].copy()
    target_train = df[(df["domain"] == target_domain) & (df["split"] == "train")].copy()
    target_test = df[(df["domain"] == target_domain) & (df["split"] == "test")].copy()

    target_fewshot = sample_fewshot_by_class(target_train, shots_per_class=8, seed=42)
    train_df = pd.concat([source_train, target_fewshot], axis=0, ignore_index=True)

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(train_df["text"], train_df["label"])
    preds = model.predict(target_test["text"])
    metrics = compute_classification_metrics(target_test["label"].tolist(), preds.tolist())

    print(f"Domain transfer: {source_domain} -> {target_domain}")
    print(f"Train rows: {len(train_df)} | Test rows: {len(target_test)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
