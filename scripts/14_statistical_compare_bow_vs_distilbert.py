from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT_DIR = Path(__file__).resolve().parents[1]
COMPARISON_DIR = ROOT_DIR / "outputs" / "runs" / "comparison_bow_vs_distilbert"


def rank_biserial_from_wilcoxon_diffs(diffs: np.ndarray) -> float:
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0

    abs_vals = np.abs(nonzero)
    ranks = pd.Series(abs_vals).rank(method="average").to_numpy()

    pos_rank_sum = ranks[nonzero > 0].sum()
    neg_rank_sum = ranks[nonzero < 0].sum()
    total_rank_sum = ranks.sum()

    return float((pos_rank_sum - neg_rank_sum) / total_rank_sum)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    means = []

    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())

    means = np.asarray(means)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    table_path = COMPARISON_DIR / "comparison_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing comparison table: {table_path}")

    df = pd.read_csv(table_path).copy()

    bow = df["bow_macro_f1_mean"].to_numpy()
    distil = df["distilbert_macro_f1_mean"].to_numpy()
    diffs = distil - bow

    stat, p_value = wilcoxon(distil, bow, alternative="greater")
    rank_biserial = rank_biserial_from_wilcoxon_diffs(diffs)

    mean_diff = float(diffs.mean())
    median_diff = float(np.median(diffs))
    ci_low, ci_high = bootstrap_mean_ci(diffs)

    summary = {
        "num_paired_transfers": int(len(df)),
        "mean_bow_macro_f1": float(bow.mean()),
        "mean_distilbert_macro_f1": float(distil.mean()),
        "mean_macro_f1_gain": mean_diff,
        "median_macro_f1_gain": median_diff,
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p_value_one_sided_distilbert_greater": float(p_value),
        "rank_biserial_effect_size": rank_biserial,
        "bootstrap_95ci_mean_gain": [ci_low, ci_high],
        "all_positive_gains": bool((diffs > 0).all()),
    }

    with open(COMPARISON_DIR / "stats_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    out_df = df[
        [
            "source_domain",
            "target_domain",
            "bow_macro_f1_mean",
            "distilbert_macro_f1_mean",
            "macro_f1_gain",
        ]
    ].copy()
    out_df["gain_rank"] = out_df["macro_f1_gain"].rank(ascending=False, method="dense")
    out_df = out_df.sort_values("macro_f1_gain", ascending=False)
    out_df.to_csv(COMPARISON_DIR / "stats_input_with_gains.csv", index=False)

    print("=== Statistical comparison ===")
    print(f"Paired transfers: {summary['num_paired_transfers']}")
    print(f"Mean BOW Macro F1       : {summary['mean_bow_macro_f1']:.4f}")
    print(f"Mean DistilBERT Macro F1: {summary['mean_distilbert_macro_f1']:.4f}")
    print(f"Mean gain               : {summary['mean_macro_f1_gain']:.4f}")
    print(f"Median gain             : {summary['median_macro_f1_gain']:.4f}")
    print(
        f"Bootstrap 95% CI gain   : "
        f"[{summary['bootstrap_95ci_mean_gain'][0]:.4f}, "
        f"{summary['bootstrap_95ci_mean_gain'][1]:.4f}]"
    )
    print(
        f"Wilcoxon p-value        : "
        f"{summary['wilcoxon_p_value_one_sided_distilbert_greater']:.6f}"
    )
    print(f"Rank-biserial effect    : {summary['rank_biserial_effect_size']:.4f}")
    print(f"All gains positive?     : {summary['all_positive_gains']}")
    print("\nSaved files:")
    print(COMPARISON_DIR / "stats_summary.json")
    print(COMPARISON_DIR / "stats_input_with_gains.csv")


if __name__ == "__main__":
    main()