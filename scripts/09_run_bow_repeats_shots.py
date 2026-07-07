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

SHOTS_PER_CLASS_LIST = [4, 8, 16]
SEEDS = list(range(42, 52))  # 10 tekrar


def sample_k_per_class(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    parts = []
    labels = sorted(df["label"].unique().tolist())

    for label in labels:
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
    shots_per_class: int,
    seed: int,
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


def summarize_results(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        results_df.groupby(["source_domain", "target_domain", "shots_per_class"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            runs=("macro_f1", "count"),
        )
        .reset_index()
        .sort_values(["shots_per_class", "source_domain", "target_domain"])
    )

    shot_summary = (
        results_df.groupby(["shots_per_class"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_std=("macro_f1", "std"),
            runs=("macro_f1", "count"),
        )
        .reset_index()
        .sort_values("shots_per_class")
    )

    return grouped, shot_summary


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = {"text", "label", "domain", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    domains = sorted(df["domain"].dropna().unique().tolist())

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_bow_repeats_shots"
    run_dir = OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_jobs = len(SHOTS_PER_CLASS_LIST) * len(SEEDS) * (len(domains) * (len(domains) - 1))
    job_idx = 0

    for shots_per_class in SHOTS_PER_CLASS_LIST:
        for seed in SEEDS:
            for source_domain in domains:
                for target_domain in domains:
                    if source_domain == target_domain:
                        continue

                    job_idx += 1
                    result = run_one_transfer(
                        df=df,
                        source_domain=source_domain,
                        target_domain=target_domain,
                        shots_per_class=shots_per_class,
                        seed=seed,
                    )
                    results.append(result)

                    print(
                        f"[{job_idx}/{total_jobs}] "
                        f"{source_domain} -> {target_domain} | "
                        f"shots={shots_per_class} | seed={seed} | "
                        f"Macro F1={result['macro_f1']:.4f}"
                    )

    results_df = pd.DataFrame(results).sort_values(
        by=["shots_per_class", "source_domain", "target_domain", "seed"]
    )
    grouped_df, shot_summary_df = summarize_results(results_df)

    raw_results_path = run_dir / "raw_results.csv"
    grouped_results_path = run_dir / "grouped_results.csv"
    shot_summary_path = run_dir / "shot_summary.csv"

    results_df.to_csv(raw_results_path, index=False)
    grouped_df.to_csv(grouped_results_path, index=False)
    shot_summary_df.to_csv(shot_summary_path, index=False)

    best_row = grouped_df.sort_values("macro_f1_mean", ascending=False).iloc[0].to_dict()
    worst_row = grouped_df.sort_values("macro_f1_mean", ascending=True).iloc[0].to_dict()

    summary = {
        "run_name": run_name,
        "dataset_path": str(DATA_PATH),
        "shots_per_class_list": SHOTS_PER_CLASS_LIST,
        "seeds": SEEDS,
        "num_domains": len(domains),
        "num_transfers_per_shot": len(domains) * (len(domains) - 1),
        "total_runs": int(len(results_df)),
        "overall_mean_accuracy": float(results_df["accuracy"].mean()),
        "overall_mean_balanced_accuracy": float(results_df["balanced_accuracy"].mean()),
        "overall_mean_macro_f1": float(results_df["macro_f1"].mean()),
        "best_transfer_by_macro_f1_mean": best_row,
        "worst_transfer_by_macro_f1_mean": worst_row,
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSaved files:")
    print(raw_results_path)
    print(grouped_results_path)
    print(shot_summary_path)
    print(run_dir / "summary.json")


if __name__ == "__main__":
    main()