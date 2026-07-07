import pandas as pd


def sample_fewshot_by_class(df: pd.DataFrame, shots_per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for label, group in df.groupby("label"):
        if len(group) < shots_per_class:
            raise ValueError(
                f"Not enough samples for label={label}. Required={shots_per_class}, found={len(group)}"
            )
        parts.append(group.sample(n=shots_per_class, random_state=seed))
    sampled = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled
