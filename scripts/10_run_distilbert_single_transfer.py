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


def main():
    cfg = load_yaml(ROOT_DIR / "configs" / "experiment_distilbert.yaml")

    exp_name = cfg["experiment_name"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    set_seed(data_cfg["seed"])

    df = pd.read_csv(ROOT_DIR / data_cfg["processed_csv"])

    source_domain = data_cfg["source_domain"]
    target_domain = data_cfg["target_domain"]
    shots_per_class = data_cfg["shots_per_class"]
    seed = data_cfg["seed"]

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

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["hf_name"],
        num_labels=2,
    )

    train_dataset = TextDataset(
        train_df["text"],
        train_df["label"],
        tokenizer,
        max_length=model_cfg["max_length"],
    )
    val_dataset = TextDataset(
        target_val["text"],
        target_val["label"],
        tokenizer,
        max_length=model_cfg["max_length"],
    )
    test_dataset = TextDataset(
        target_test["text"],
        target_test["label"],
        tokenizer,
        max_length=model_cfg["max_length"],
    )

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{exp_name}"
    run_dir = ROOT_DIR / out_cfg["save_dir"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(run_dir / "hf_outputs"),
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["train_batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        num_train_epochs=train_cfg["num_train_epochs"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_metrics = trainer.evaluate(eval_dataset=test_dataset)
    preds_output = trainer.predict(test_dataset)
    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_true = target_test["label"].to_numpy()

    result = {
        "experiment_name": exp_name,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "shots_per_class": shots_per_class,
        "seed": seed,
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

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    pd.DataFrame([result]).to_csv(run_dir / "summary.csv", index=False)

    print("Run completed:", run_name)
    print("Device:", result["device"])
    print("Accuracy:", f"{result['accuracy']:.4f}")
    print("Balanced Accuracy:", f"{result['balanced_accuracy']:.4f}")
    print("Macro F1:", f"{result['macro_f1']:.4f}")
    print("Saved to:", run_dir / "summary.json")


if __name__ == "__main__":
    main()