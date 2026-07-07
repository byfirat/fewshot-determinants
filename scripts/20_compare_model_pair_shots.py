from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from fewshot_determinants.utils.io import load_yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to comparison YAML config, relative to repo root.",
    )
    return parser.parse_args()


def load_grouped_results(path_str: str, model_name: str, metric_column: str) -> pd.DataFrame:
    path = ROOT_DIR / path_str
    if not path.exists():
        raise FileNotFoundError(f"Grouped results file not found: {path}")

    df = pd.read_csv(path)

    required_cols = {
        "source_domain",
        "target_domain",
        "shots_per_class",
        metric_column,
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in grouped results ({model_name}): {sorted(missing)}"
        )

    out = df[
        ["source_domain", "target_domain", "shots_per_class", metric_column]
    ].copy()
    out = out.rename(columns={metric_column: f"{model_name}_{metric_column}"})
    return out


def validate_merge(merged_df: pd.DataFrame, model_a_name: str, model_b_name: str) -> None:
    if merged_df.empty:
        raise ValueError("Merged comparison table is empty.")

    if merged_df.isna().any().any():
        raise ValueError(
            f"Merged table contains NaN values for {model_a_name} vs {model_b_name}."
        )

    counts = (
        merged_df.groupby("shots_per_class")
        .size()
        .reset_index(name="n_pairs")
        .sort_values("shots_per_class")
    )

    if (counts["n_pairs"] == 0).any():
        raise ValueError("At least one shot level has zero paired transfers.")


def bootstrap_mean_ci(
    values: np.ndarray,
    num_resamples: int,
    random_seed: int,
    ci_level: float,
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(random_seed)
    means = np.empty(num_resamples, dtype=float)

    for i in range(num_resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = np.mean(sample)

    alpha = 1.0 - ci_level
    low = np.quantile(means, alpha / 2.0)
    high = np.quantile(means, 1.0 - alpha / 2.0)

    return float(low), float(high)


def rank_biserial_effect(gains: np.ndarray) -> float:
    non_zero = gains[gains != 0]
    if len(non_zero) == 0:
        return 0.0

    ranks = rankdata(np.abs(non_zero), method="average")
    pos_rank_sum = float(ranks[non_zero > 0].sum())
    neg_rank_sum = float(ranks[non_zero < 0].sum())
    denom = len(non_zero) * (len(non_zero) + 1) / 2.0

    return float((pos_rank_sum - neg_rank_sum) / denom)


def compare_one_shot(
    shot_df: pd.DataFrame,
    model_a_name: str,
    model_b_name: str,
    metric_column: str,
    num_resamples: int,
    random_seed: int,
    ci_level: float,
) -> dict:
    col_a = f"{model_a_name}_{metric_column}"
    col_b = f"{model_b_name}_{metric_column}"

    gains = (shot_df[col_b] - shot_df[col_a]).to_numpy(dtype=float)

    improved = int(np.sum(gains > 0))
    worsened = int(np.sum(gains < 0))
    tied = int(np.sum(gains == 0))
    all_gains_positive = bool(np.all(gains > 0))

    ci_low, ci_high = bootstrap_mean_ci(
        values=gains,
        num_resamples=num_resamples,
        random_seed=random_seed,
        ci_level=ci_level,
    )

    non_zero = gains[gains != 0]
    if len(non_zero) == 0:
        p_value = float("nan")
    else:
        p_value = float(wilcoxon(gains, zero_method="wilcox").pvalue)

    effect_size = rank_biserial_effect(gains)

    return {
        "shots_per_class": int(shot_df["shots_per_class"].iloc[0]),
        "n_paired_transfers": int(len(shot_df)),
        f"{model_a_name}_{metric_column}_mean": float(shot_df[col_a].mean()),
        f"{model_b_name}_{metric_column}_mean": float(shot_df[col_b].mean()),
        "mean_gain": float(np.mean(gains)),
        "median_gain": float(np.median(gains)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "wilcoxon_p_value": p_value,
        "rank_biserial_effect_size": effect_size,
        "improved_transfers": improved,
        "worsened_transfers": worsened,
        "tied_transfers": tied,
        "all_gains_positive": all_gains_positive,
    }


def build_shot_summary_comparison(
    merged_df: pd.DataFrame,
    model_a_name: str,
    model_b_name: str,
    metric_column: str,
) -> pd.DataFrame:
    col_a = f"{model_a_name}_{metric_column}"
    col_b = f"{model_b_name}_{metric_column}"

    out = (
        merged_df.groupby("shots_per_class")
        .agg(
            model_a_mean=(col_a, "mean"),
            model_b_mean=(col_b, "mean"),
            mean_gain=("gain", "mean"),
            median_gain=("gain", "median"),
            n_pairs=("gain", "count"),
        )
        .reset_index()
        .sort_values("shots_per_class")
    )

    out = out.rename(
        columns={
            "model_a_mean": f"{model_a_name}_{metric_column}_mean",
            "model_b_mean": f"{model_b_name}_{metric_column}_mean",
        }
    )
    return out


def main():
    args = parse_args()
    cfg = load_yaml(ROOT_DIR / args.config)

    inputs_cfg = cfg["inputs"]
    comparison_cfg = cfg["comparison"]
    bootstrap_cfg = cfg["bootstrap"]
    output_cfg = cfg["output"]

    model_a_name = inputs_cfg["model_a_name"]
    model_b_name = inputs_cfg["model_b_name"]
    metric_column = comparison_cfg["metric_column"]
    shots_per_class_list = [int(x) for x in comparison_cfg["shots_per_class_list"]]

    model_a_df = load_grouped_results(
        path_str=inputs_cfg["model_a_grouped_results_csv"],
        model_name=model_a_name,
        metric_column=metric_column,
    )
    model_b_df = load_grouped_results(
        path_str=inputs_cfg["model_b_grouped_results_csv"],
        model_name=model_b_name,
        metric_column=metric_column,
    )

    model_a_df = model_a_df[model_a_df["shots_per_class"].isin(shots_per_class_list)].copy()
    model_b_df = model_b_df[model_b_df["shots_per_class"].isin(shots_per_class_list)].copy()

    merged_df = model_a_df.merge(
        model_b_df,
        on=["source_domain", "target_domain", "shots_per_class"],
        how="inner",
        validate="one_to_one",
    )

    validate_merge(merged_df, model_a_name=model_a_name, model_b_name=model_b_name)

    col_a = f"{model_a_name}_{metric_column}"
    col_b = f"{model_b_name}_{metric_column}"
    merged_df["gain"] = merged_df[col_b] - merged_df[col_a]
    merged_df = merged_df.sort_values(
        ["shots_per_class", "source_domain", "target_domain"]
    ).reset_index(drop=True)

    shot_stats_rows = []
    for shots_per_class in shots_per_class_list:
        shot_df = merged_df[merged_df["shots_per_class"] == shots_per_class].copy()
        if shot_df.empty:
            raise ValueError(f"No paired rows found for shots_per_class={shots_per_class}")

        row = compare_one_shot(
            shot_df=shot_df,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            metric_column=metric_column,
            num_resamples=int(bootstrap_cfg["num_resamples"]),
            random_seed=int(bootstrap_cfg["random_seed"]),
            ci_level=float(bootstrap_cfg["ci_level"]),
        )
        shot_stats_rows.append(row)

    statistical_df = pd.DataFrame(shot_stats_rows).sort_values("shots_per_class")
    shot_summary_df = build_shot_summary_comparison(
        merged_df=merged_df,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        metric_column=metric_column,
    )

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg['comparison_name']}"
    run_dir = ROOT_DIR / output_cfg["save_dir"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(run_dir / "paired_transfer_comparison.csv", index=False)
    shot_summary_df.to_csv(run_dir / "shot_summary_comparison.csv", index=False)
    statistical_df.to_csv(run_dir / "statistical_comparison_by_shot.csv", index=False)

    best_shot_row = statistical_df.sort_values("mean_gain", ascending=False).iloc[0].to_dict()
    summary = {
        "run_name": run_name,
        "model_a_name": model_a_name,
        "model_b_name": model_b_name,
        "metric_column": metric_column,
        "shots_per_class_list": shots_per_class_list,
        "num_paired_rows": int(len(merged_df)),
        "overall_model_a_mean": float(merged_df[col_a].mean()),
        "overall_model_b_mean": float(merged_df[col_b].mean()),
        "overall_mean_gain": float(merged_df["gain"].mean()),
        "best_shot_by_mean_gain": best_shot_row,
        "output_files": [
            "paired_transfer_comparison.csv",
            "shot_summary_comparison.csv",
            "statistical_comparison_by_shot.csv",
            "summary.json",
        ],
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Comparison complete ===")
    print(f"{model_a_name} vs {model_b_name}")
    print(f"Overall mean gain ({model_b_name} - {model_a_name}): {summary['overall_mean_gain']:.4f}")
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()