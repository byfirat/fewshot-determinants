from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "outputs" / "runs"

# Gerekirse bunları sen kendi klasör adına göre güncelle
BOW_RUN_DIR = OUTPUTS_DIR / "20260422_230251_bow_repeats_shots"
DISTILBERT_RUN_DIR = OUTPUTS_DIR / "20260423_031006_distilbert_transfer_matrix"


def main():
    bow_path = BOW_RUN_DIR / "grouped_results.csv"
    distilbert_path = DISTILBERT_RUN_DIR / "grouped_results.csv"

    if not bow_path.exists():
        raise FileNotFoundError(f"BOW grouped_results not found: {bow_path}")
    if not distilbert_path.exists():
        raise FileNotFoundError(f"DistilBERT grouped_results not found: {distilbert_path}")

    bow_df = pd.read_csv(bow_path).copy()
    distilbert_df = pd.read_csv(distilbert_path).copy()

    # BOW tarafında sadece 8-shot sonuçlarını alıyoruz ki adil karşılaştırma olsun
    bow_df = bow_df[bow_df["shots_per_class"] == 8].copy()

    bow_keep = bow_df[
        [
            "source_domain",
            "target_domain",
            "shots_per_class",
            "macro_f1_mean",
            "macro_f1_std",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "runs",
        ]
    ].rename(
        columns={
            "macro_f1_mean": "bow_macro_f1_mean",
            "macro_f1_std": "bow_macro_f1_std",
            "accuracy_mean": "bow_accuracy_mean",
            "balanced_accuracy_mean": "bow_balanced_accuracy_mean",
            "runs": "bow_runs",
        }
    )

    distilbert_keep = distilbert_df[
        [
            "source_domain",
            "target_domain",
            "shots_per_class",
            "macro_f1_mean",
            "macro_f1_std",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "runs",
        ]
    ].rename(
        columns={
            "macro_f1_mean": "distilbert_macro_f1_mean",
            "macro_f1_std": "distilbert_macro_f1_std",
            "accuracy_mean": "distilbert_accuracy_mean",
            "balanced_accuracy_mean": "distilbert_balanced_accuracy_mean",
            "runs": "distilbert_runs",
        }
    )

    merged = bow_keep.merge(
        distilbert_keep,
        on=["source_domain", "target_domain", "shots_per_class"],
        how="inner",
    )

    merged["macro_f1_gain"] = (
        merged["distilbert_macro_f1_mean"] - merged["bow_macro_f1_mean"]
    )
    merged["accuracy_gain"] = (
        merged["distilbert_accuracy_mean"] - merged["bow_accuracy_mean"]
    )
    merged["balanced_accuracy_gain"] = (
        merged["distilbert_balanced_accuracy_mean"]
        - merged["bow_balanced_accuracy_mean"]
    )

    merged = merged.sort_values("macro_f1_gain", ascending=False).reset_index(drop=True)

    comparison_dir = OUTPUTS_DIR / "comparison_bow_vs_distilbert"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(comparison_dir / "comparison_table.csv", index=False)

    summary = {
        "num_transfers": int(len(merged)),
        "mean_bow_macro_f1": float(merged["bow_macro_f1_mean"].mean()),
        "mean_distilbert_macro_f1": float(merged["distilbert_macro_f1_mean"].mean()),
        "mean_macro_f1_gain": float(merged["macro_f1_gain"].mean()),
        "best_gain_transfer": merged.sort_values("macro_f1_gain", ascending=False)
        .iloc[0]
        .to_dict(),
        "worst_gain_transfer": merged.sort_values("macro_f1_gain", ascending=True)
        .iloc[0]
        .to_dict(),
    }

    with open(comparison_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Comparison summary ===")
    print(f"Transfers compared: {summary['num_transfers']}")
    print(f"Mean BOW Macro F1       : {summary['mean_bow_macro_f1']:.4f}")
    print(f"Mean DistilBERT Macro F1: {summary['mean_distilbert_macro_f1']:.4f}")
    print(f"Mean Macro F1 Gain      : {summary['mean_macro_f1_gain']:.4f}")

    print("\nTop 5 gains by Macro F1:")
    print(
        merged[
            [
                "source_domain",
                "target_domain",
                "bow_macro_f1_mean",
                "distilbert_macro_f1_mean",
                "macro_f1_gain",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print("\nBottom 5 gains by Macro F1:")
    print(
        merged[
            [
                "source_domain",
                "target_domain",
                "bow_macro_f1_mean",
                "distilbert_macro_f1_mean",
                "macro_f1_gain",
            ]
        ]
        .tail(5)
        .to_string(index=False)
    )

    print("\nSaved files:")
    print(comparison_dir / "comparison_table.csv")
    print(comparison_dir / "summary.json")


if __name__ == "__main__":
    main()