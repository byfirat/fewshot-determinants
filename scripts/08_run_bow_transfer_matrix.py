from __future__ import annotations

from datetime import datetime
from pathlib import Path

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "amazon_multidomain_sentiment_rawtext.csv"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "runs"


def sample_k_per_class(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    parts = []
    for label in sorted(df["label"].unique()):
        label_df = df[df["label"] == label].copy()
        if len(label_df) < k:
            raise ValueError(
                f"Not enough rows for label={label}. Needed {k}, found {len(label_df)}."
            )
        parts.append(label_df.sample(n=k, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def run_one_transfer(
    df: pd.DataFrame,
    source_domain: str,
    target_domain: str,
    shots_per_class: int = 8,
    seed: int = 42,
) -> dict:
    source_train = df[
        (df["domain"] == source_domain) & (df["split"] == "train")
    ].copy()

    target_train = df[
        (df["domain"] == target_domain) & (df["split"] == "train")
    ].copy()

    target_test = df[
        (df["domain"] == target_domain) & (df["split"] == "test")
    ].copy()

    target_support = sample_k_per_class(target_train, k=shots_per_class, seed=seed)

    train_df = pd.concat([source_train, target_support], ignore_index=True)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )

    x_train = vectorizer.fit_transform(train_df["text"])
    x_test = vectorizer.transform(target_test["text"])

    y_train = train_df["label"]
    y_test = target_test["label"]

    clf = LogisticRegression(
        max_iter=2000,
        random_state=seed,
        class_weight="balanced",
        solver="liblinear",
    )
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    return {
        "source_domain": source_domain,
        "target_domain": target_domain,
        "shots_per_class": shots_per_class,
        "seed": seed,
        "source_train_rows": int(len(source_train)),
        "target_support_rows": int(len(target_support)),
        "total_train_rows": int(len(train_df)),
        "test_rows": int(len(target_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = {"text", "label", "domain", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    domains = sorted(df["domain"].dropna().unique().tolist())

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_bow_transfer_matrix"
    run_dir = OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for source_domain in domains:
        for target_domain in domains:
            if source_domain == target_domain:
                continue

            result = run_one_transfer(
                df=df,
                source_domain=source_domain,
                target_domain=target_domain,
                shots_per_class=8,
                seed=42,
            )
            results.append(result)

            print(
                f"{source_domain} -> {target_domain} | "
                f"Macro F1={result['macro_f1']:.4f} | "
                f"Accuracy={result['accuracy']:.4f} | "
                f"Balanced Acc={result['balanced_accuracy']:.4f}"
            )

    results_df = pd.DataFrame(results).sort_values(
        by=["source_domain", "target_domain"]
    )

    results_path = run_dir / "transfer_results.csv"
    results_df.to_csv(results_path, index=False)

    macro_f1_matrix = results_df.pivot(
        index="source_domain", columns="target_domain", values="macro_f1"
    )
    accuracy_matrix = results_df.pivot(
        index="source_domain", columns="target_domain", values="accuracy"
    )
    balanced_acc_matrix = results_df.pivot(
        index="source_domain", columns="target_domain", values="balanced_accuracy"
    )

    macro_f1_matrix.to_csv(run_dir / "macro_f1_matrix.csv")
    accuracy_matrix.to_csv(run_dir / "accuracy_matrix.csv")
    balanced_acc_matrix.to_csv(run_dir / "balanced_accuracy_matrix.csv")

    summary = {
        "run_name": run_name,
        "dataset_path": str(DATA_PATH),
        "shots_per_class": 8,
        "seed": 42,
        "num_domains": len(domains),
        "num_transfers": int(len(results_df)),
        "mean_accuracy": float(results_df["accuracy"].mean()),
        "mean_balanced_accuracy": float(results_df["balanced_accuracy"].mean()),
        "mean_macro_f1": float(results_df["macro_f1"].mean()),
        "best_transfer_by_macro_f1": results_df.sort_values(
            "macro_f1", ascending=False
        ).iloc[0].to_dict(),
        "worst_transfer_by_macro_f1": results_df.sort_values(
            "macro_f1", ascending=True
        ).iloc[0].to_dict(),
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSaved files:")
    print(results_path)
    print(run_dir / "macro_f1_matrix.csv")
    print(run_dir / "accuracy_matrix.csv")
    print(run_dir / "balanced_accuracy_matrix.csv")
    print(run_dir / "summary.json")


if __name__ == "__main__":
    main()