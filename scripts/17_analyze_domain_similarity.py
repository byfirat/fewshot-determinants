from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fewshot_determinants.utils.io import load_yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configs" / "analysis_domain_similarity.yaml"


def tokenize(text: str, min_token_length: int) -> list[str]:
    tokens = re.findall(r"[A-Za-z']+", str(text).lower())
    return [tok for tok in tokens if len(tok) >= min_token_length]


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def js_similarity_from_counts(
    counts_a: Counter,
    counts_b: Counter,
    token_top_k: int,
) -> float:
    combined = counts_a + counts_b
    vocab = [token for token, _ in combined.most_common(token_top_k)]

    if not vocab:
        return float("nan")

    vec_a = np.array([counts_a.get(tok, 0) for tok in vocab], dtype=float)
    vec_b = np.array([counts_b.get(tok, 0) for tok in vocab], dtype=float)

    if vec_a.sum() == 0 or vec_b.sum() == 0:
        return float("nan")

    p = vec_a / vec_a.sum()
    q = vec_b / vec_b.sum()
    m = 0.5 * (p + q)

    js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    js_sim = 1.0 - js_div  # base-2 KL => JS divergence is in [0, 1]

    return float(js_sim)


def jaccard_top_tokens(
    counts_a: Counter,
    counts_b: Counter,
    token_top_k: int,
) -> float:
    top_a = {token for token, _ in counts_a.most_common(token_top_k)}
    top_b = {token for token, _ in counts_b.most_common(token_top_k)}

    union = top_a | top_b
    if not union:
        return float("nan")

    inter = top_a & top_b
    return float(len(inter) / len(union))


def build_domain_counters(
    df: pd.DataFrame,
    domains: list[str],
    min_token_length: int,
) -> dict[str, Counter]:
    counters: dict[str, Counter] = {}

    for domain in domains:
        domain_df = df[df["domain"] == domain].copy()
        counter: Counter = Counter()

        for text in domain_df["text"].astype(str).tolist():
            counter.update(tokenize(text, min_token_length=min_token_length))

        counters[domain] = counter

    return counters


def build_tfidf_centroid_similarities(
    df: pd.DataFrame,
    domains: list[str],
    max_features: int,
) -> dict[tuple[str, str], float]:
    texts = df["text"].astype(str).tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
    )
    x = vectorizer.fit_transform(texts)

    centroid_by_domain: dict[str, np.ndarray] = {}
    for domain in domains:
        idx = df.index[df["domain"] == domain].tolist()
        domain_matrix = x[idx]
        centroid = np.asarray(domain_matrix.mean(axis=0)).ravel()
        centroid_by_domain[domain] = centroid

    sims: dict[tuple[str, str], float] = {}
    for source_domain in domains:
        for target_domain in domains:
            if source_domain == target_domain:
                continue

            a = centroid_by_domain[source_domain].reshape(1, -1)
            b = centroid_by_domain[target_domain].reshape(1, -1)
            sims[(source_domain, target_domain)] = float(cosine_similarity(a, b)[0, 0])

    return sims


def build_similarity_table(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    sim_cfg = cfg["similarity"]

    data_path = ROOT_DIR / data_cfg["processed_csv"]
    df = pd.read_csv(data_path)

    required_cols = {"text", "label", "domain", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")

    split_name = data_cfg["split_for_similarity"]
    split_df = df[df["split"] == split_name].copy()
    if split_df.empty:
        raise ValueError(f"No rows found for split='{split_name}'")

    domains = sorted(split_df["domain"].dropna().unique().tolist())

    min_token_length = int(sim_cfg["min_token_length"])
    token_top_k = int(sim_cfg["token_top_k"])
    tfidf_max_features = int(sim_cfg["tfidf_max_features"])

    domain_counters = build_domain_counters(
        df=split_df,
        domains=domains,
        min_token_length=min_token_length,
    )

    tfidf_sims = build_tfidf_centroid_similarities(
        df=split_df.reset_index(drop=True),
        domains=domains,
        max_features=tfidf_max_features,
    )

    label_rate_by_domain: dict[str, float] = {}
    doc_count_by_domain: dict[str, int] = {}
    avg_text_len_by_domain: dict[str, float] = {}

    for domain in domains:
        domain_df = split_df[split_df["domain"] == domain].copy()
        label_rate_by_domain[domain] = float(domain_df["label"].mean())
        doc_count_by_domain[domain] = int(len(domain_df))
        avg_text_len_by_domain[domain] = float(
            domain_df["text"].astype(str).map(lambda x: len(x.split())).mean()
        )

    rows = []
    for source_domain in domains:
        for target_domain in domains:
            if source_domain == target_domain:
                continue

            counts_a = domain_counters[source_domain]
            counts_b = domain_counters[target_domain]

            js_sim = js_similarity_from_counts(
                counts_a=counts_a,
                counts_b=counts_b,
                token_top_k=token_top_k,
            )
            jaccard_sim = jaccard_top_tokens(
                counts_a=counts_a,
                counts_b=counts_b,
                token_top_k=token_top_k,
            )
            tfidf_sim = tfidf_sims[(source_domain, target_domain)]

            label_prior_similarity = 1.0 - abs(
                label_rate_by_domain[source_domain] - label_rate_by_domain[target_domain]
            )

            avg_len_ratio = min(
                avg_text_len_by_domain[source_domain],
                avg_text_len_by_domain[target_domain],
            ) / max(
                avg_text_len_by_domain[source_domain],
                avg_text_len_by_domain[target_domain],
            )

            rows.append(
                {
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "tfidf_centroid_cosine": float(tfidf_sim),
                    "token_jaccard_topk": float(jaccard_sim),
                    "js_unigram_similarity": float(js_sim),
                    "label_prior_similarity": float(label_prior_similarity),
                    "avg_doc_length_ratio": float(avg_len_ratio),
                    "source_doc_count": doc_count_by_domain[source_domain],
                    "target_doc_count": doc_count_by_domain[target_domain],
                    "source_positive_rate": label_rate_by_domain[source_domain],
                    "target_positive_rate": label_rate_by_domain[target_domain],
                    "source_avg_doc_len": avg_text_len_by_domain[source_domain],
                    "target_avg_doc_len": avg_text_len_by_domain[target_domain],
                }
            )

    return pd.DataFrame(rows).sort_values(["source_domain", "target_domain"]).reset_index(
        drop=True
    )


def load_grouped_results(path_str: str, model_name: str) -> pd.DataFrame:
    path = ROOT_DIR / path_str
    df = pd.read_csv(path)

    required_cols = {
        "source_domain",
        "target_domain",
        "shots_per_class",
        "macro_f1_mean",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in grouped results ({model_name}): {sorted(missing)}"
        )

    out = df.copy()
    out["model_name"] = model_name
    return out


def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float]:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)

    if len(x_arr) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")

    if np.isclose(np.std(x_arr), 0.0) or np.isclose(np.std(y_arr), 0.0):
        return float("nan"), float("nan"), float("nan"), float("nan")

    pearson_r, pearson_p = pearsonr(x_arr, y_arr)
    spearman_rho, spearman_p = spearmanr(x_arr, y_arr)

    return (
        float(pearson_r),
        float(pearson_p),
        float(spearman_rho),
        float(spearman_p),
    )


def build_correlation_table(merged_df: pd.DataFrame) -> pd.DataFrame:
    similarity_metrics = [
        "tfidf_centroid_cosine",
        "token_jaccard_topk",
        "js_unigram_similarity",
        "label_prior_similarity",
        "avg_doc_length_ratio",
    ]

    rows = []

    for model_name in sorted(merged_df["model_name"].unique().tolist()):
        model_df = merged_df[merged_df["model_name"] == model_name].copy()

        for shots_per_class in sorted(model_df["shots_per_class"].unique().tolist()):
            sub = model_df[model_df["shots_per_class"] == shots_per_class].copy()

            for metric_name in similarity_metrics:
                pearson_r, pearson_p, spearman_rho, spearman_p = safe_corr(
                    sub[metric_name],
                    sub["macro_f1_mean"],
                )

                rows.append(
                    {
                        "model_name": model_name,
                        "shots_per_class": int(shots_per_class),
                        "metric_name": metric_name,
                        "n_transfers": int(len(sub)),
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_rho": spearman_rho,
                        "spearman_p": spearman_p,
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["model_name", "shots_per_class", "metric_name"]
    ).reset_index(drop=True)


def build_pair_average_table(merged_df: pd.DataFrame) -> pd.DataFrame:
    df = merged_df.copy()

    df["pair_key"] = df.apply(
        lambda row: " || ".join(sorted([row["source_domain"], row["target_domain"]])),
        axis=1,
    )

    pair_avg = (
        df.groupby(
            [
                "model_name",
                "shots_per_class",
                "pair_key",
                "tfidf_centroid_cosine",
                "token_jaccard_topk",
                "js_unigram_similarity",
                "label_prior_similarity",
                "avg_doc_length_ratio",
            ],
            as_index=False,
        )
        .agg(
            macro_f1_pair_avg=("macro_f1_mean", "mean"),
            directional_count=("macro_f1_mean", "count"),
        )
        .sort_values(["model_name", "shots_per_class", "pair_key"])
        .reset_index(drop=True)
    )

    return pair_avg


def build_pair_correlation_table(pair_avg_df: pd.DataFrame) -> pd.DataFrame:
    similarity_metrics = [
        "tfidf_centroid_cosine",
        "token_jaccard_topk",
        "js_unigram_similarity",
        "label_prior_similarity",
        "avg_doc_length_ratio",
    ]

    rows = []

    for model_name in sorted(pair_avg_df["model_name"].unique().tolist()):
        model_df = pair_avg_df[pair_avg_df["model_name"] == model_name].copy()

        for shots_per_class in sorted(model_df["shots_per_class"].unique().tolist()):
            sub = model_df[model_df["shots_per_class"] == shots_per_class].copy()

            for metric_name in similarity_metrics:
                pearson_r, pearson_p, spearman_rho, spearman_p = safe_corr(
                    sub[metric_name],
                    sub["macro_f1_pair_avg"],
                )

                rows.append(
                    {
                        "model_name": model_name,
                        "shots_per_class": int(shots_per_class),
                        "metric_name": metric_name,
                        "n_pairs": int(len(sub)),
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_rho": spearman_rho,
                        "spearman_p": spearman_p,
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["model_name", "shots_per_class", "metric_name"]
    ).reset_index(drop=True)

def main():

    cfg = load_yaml(CONFIG_PATH)

    output_cfg = cfg["output"]

    inputs_cfg = cfg["inputs"]

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg['analysis_name']}"

    run_dir = ROOT_DIR / output_cfg["save_dir"] / run_name

    run_dir.mkdir(parents=True, exist_ok=True)

    similarity_df = build_similarity_table(cfg)

    # Yeni yapı:

    # inputs:

    #   grouped_results:

    #     bow_tfidf_logreg: outputs/runs/.../grouped_results.csv

    #     distilbert: outputs/runs/.../grouped_results.csv

    #     roberta_base: outputs/runs/.../grouped_results.csv

    if "grouped_results" not in inputs_cfg:

        raise KeyError(

            "Config dosyasında inputs.grouped_results bulunamadı. "

            "analysis_domain_similarity.yaml dosyasını yeni grouped_results yapısına göre güncelle."

        )

    grouped_results_cfg = inputs_cfg["grouped_results"]

    model_frames = []

    for model_name, path_str in grouped_results_cfg.items():

        model_frames.append(

            load_grouped_results(

                path_str=path_str,

                model_name=model_name,

            )

        )

    combined_results_df = pd.concat(model_frames, ignore_index=True)

    merged_df = combined_results_df.merge(

        similarity_df,

        on=["source_domain", "target_domain"],

        how="left",

        validate="many_to_one",

    )

    if merged_df.isna().any().any():

        raise ValueError("Merged analysis table contains NaN values. Check merge inputs.")

    corr_df = build_correlation_table(merged_df)

    pair_avg_df = build_pair_average_table(merged_df)

    pair_corr_df = build_pair_correlation_table(pair_avg_df)

    similarity_df.to_csv(run_dir / "domain_similarity_pairs.csv", index=False)

    merged_df.to_csv(run_dir / "merged_similarity_performance.csv", index=False)

    corr_df.to_csv(run_dir / "similarity_correlations_directional.csv", index=False)

    pair_avg_df.to_csv(run_dir / "pair_average_performance.csv", index=False)

    pair_corr_df.to_csv(run_dir / "similarity_correlations_pair_average.csv", index=False)

    directional_best = (

        corr_df.dropna(subset=["spearman_rho"])

        .sort_values("spearman_rho", ascending=False)

        .iloc[0]

        .to_dict()

    )

    pair_best = (

        pair_corr_df.dropna(subset=["spearman_rho"])

        .sort_values("spearman_rho", ascending=False)

        .iloc[0]

        .to_dict()

    )

    summary = {

        "run_name": run_name,

        "models_included": sorted(merged_df["model_name"].unique().tolist()),

        "num_directional_rows": int(len(merged_df)),

        "num_pair_average_rows": int(len(pair_avg_df)),

        "directional_best_spearman_row": directional_best,

        "pair_average_best_spearman_row": pair_best,

        "output_files": [

            "domain_similarity_pairs.csv",

            "merged_similarity_performance.csv",

            "similarity_correlations_directional.csv",

            "pair_average_performance.csv",

            "similarity_correlations_pair_average.csv",

            "summary.json",

        ],

    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:

        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Domain similarity analysis complete ===")

    print(f"Saved to: {run_dir}")

    print("\nModels included:")

    for model_name in summary["models_included"]:

        print(f"- {model_name}")

    print(f"\nDirectional rows: {summary['num_directional_rows']}")

    print(f"Pair-average rows: {summary['num_pair_average_rows']}")

if __name__ == "__main__":
    main()