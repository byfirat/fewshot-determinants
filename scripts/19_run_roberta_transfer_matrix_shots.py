from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from fewshot_determinants.utils.io import load_yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configs" / "experiment_roberta_transfer_matrix_shots.yaml"
SEEDS = [42, 43, 44, 45, 46]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def run_one_transfer(
    cfg: dict,
    source_domain: str,
    target_domain: str,
    shots_per_class: int,
    seed: int,
    parent_run_dir: Path,
) -> dict:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    set_seed(seed)

    df = pd.read_csv(ROOT_DIR / data_cfg["processed_csv"])

    source_train = df[
        (df["domain"] == source_domain) & (df["split"] == "train")
    ].copy()

    target_train = df[
        (df["domain"] == target_domain) & (df["split"] == "train")
    ].copy()

    target_val = df[
        (df["domain"] == target_domain) & (df["split"] == "validation")
    ].copy()

    target_test = df[
        (df["domain"] == target_domain) & (df["split"] == "test")
    ].copy()

    target_support = sample_k_per_class(target_train, k=shots_per_class, seed=seed)
    train_df = pd.concat([source_train, target_support], ignore_index=True)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_name"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["hf_name"],
        num_labels=2,
    )

    train_dataset = TextDataset(
        train_df["text"],
        train_df["label"],
        tokenizer,
        max_length=int(model_cfg["max_length"]),
    )
    val_dataset = TextDataset(
        target_val["text"],
        target_val["label"],
        tokenizer,
        max_length=int(model_cfg["max_length"]),
    )
    test_dataset = TextDataset(
        target_test["text"],
        target_test["label"],
        tokenizer,
        max_length=int(model_cfg["max_length"]),
    )

    transfer_name = (
        f"{source_domain.replace(' ', '_')}_to_{target_domain.replace(' ', '_')}"
        f"_{shots_per_class}shot_seed_{seed}"
    )
    transfer_run_dir = parent_run_dir / transfer_name
    transfer_run_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(transfer_run_dir / "hf_outputs"),
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["eval_batch_size"]),
        num_train_epochs=int(train_cfg["num_train_epochs"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    preds_output = trainer.predict(test_dataset)
    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_true = target_test["label"].to_numpy()

    result = {
        "seed": seed,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "shots_per_class": shots_per_class,
        "source_train_rows": int(len(source_train)),
        "target_support_rows": int(len(target_support)),
        "total_train_rows": int(len(train_df)),
        "validation_rows": int(len(target_val)),
        "test_rows": int(len(target_test)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_name": model_cfg["hf_name"],
    }

    with open(transfer_run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def summarize_results(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped_df = (
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

    shot_summary_df = (
        results_df.groupby(["shots_per_class"])
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
        .sort_values("shots_per_class")
    )

    return grouped_df, shot_summary_df


def save_macro_f1_matrices(grouped_df: pd.DataFrame, run_dir: Path) -> None:
    for shots_per_class in sorted(grouped_df["shots_per_class"].unique().tolist()):
        shot_df = grouped_df[grouped_df["shots_per_class"] == shots_per_class].copy()

        matrix_df = shot_df.pivot(
            index="source_domain",
            columns="target_domain",
            values="macro_f1_mean",
        )
        matrix_df.to_csv(run_dir / f"macro_f1_matrix_{shots_per_class}shot.csv")


def main():
    cfg = load_yaml(CONFIG_PATH)
    data_cfg = cfg["data"]
    output_cfg = cfg["output"]

    data_path = ROOT_DIR / data_cfg["processed_csv"]
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    required_cols = {"text", "label", "domain", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    domains = sorted(df["domain"].dropna().unique().tolist())
    shots_per_class_list = [int(x) for x in data_cfg["shots_per_class_list"]]

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg['experiment_name']}"
    parent_run_dir = ROOT_DIR / output_cfg["save_dir"] / run_name
    parent_run_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_jobs = len(SEEDS) * len(shots_per_class_list) * len(domains) * (len(domains) - 1)
    job_idx = 0

    for shots_per_class in shots_per_class_list:
        for seed in SEEDS:
            for source_domain in domains:
                for target_domain in domains:
                    if source_domain == target_domain:
                        continue

                    job_idx += 1
                    print(
                        f"\n[{job_idx}/{total_jobs}] "
                        f"shots={shots_per_class} | seed={seed} | "
                        f"{source_domain} -> {target_domain}"
                    )

                    result = run_one_transfer(
                        cfg=cfg,
                        source_domain=source_domain,
                        target_domain=target_domain,
                        shots_per_class=shots_per_class,
                        seed=seed,
                        parent_run_dir=parent_run_dir,
                    )
                    all_results.append(result)

                    print(
                        f"Done | Accuracy={result['accuracy']:.4f} | "
                        f"Balanced Accuracy={result['balanced_accuracy']:.4f} | "
                        f"Macro F1={result['macro_f1']:.4f}"
                    )

    results_df = pd.DataFrame(all_results).sort_values(
        ["shots_per_class", "source_domain", "target_domain", "seed"]
    )
    grouped_df, shot_summary_df = summarize_results(results_df)

    results_df.to_csv(parent_run_dir / "raw_results.csv", index=False)
    grouped_df.to_csv(parent_run_dir / "grouped_results.csv", index=False)
    shot_summary_df.to_csv(parent_run_dir / "shot_summary.csv", index=False)
    save_macro_f1_matrices(grouped_df, parent_run_dir)

    best_row = grouped_df.sort_values("macro_f1_mean", ascending=False).iloc[0].to_dict()
    worst_row = grouped_df.sort_values("macro_f1_mean", ascending=True).iloc[0].to_dict()

    summary = {
        "run_name": run_name,
        "dataset_path": str(data_path),
        "model_name": cfg["model"]["hf_name"],
        "shots_per_class_list": shots_per_class_list,
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

    with open(parent_run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Final summary ===")
    print(f"Model: {summary['model_name']}")
    print(f"Overall Macro F1 mean: {summary['overall_mean_macro_f1']:.4f}")
    print(f"Saved to: {parent_run_dir}")


if __name__ == "__main__":
    main()