import pandas as pd

from fewshot_determinants.data.sampler import sample_fewshot_by_class


def test_sample_fewshot_by_class_balanced():
    df = pd.DataFrame({
        "text": [f"t{i}" for i in range(20)],
        "label": [0] * 10 + [1] * 10,
        "domain": ["books"] * 20,
        "split": ["train"] * 20,
    })
    sampled = sample_fewshot_by_class(df, shots_per_class=4, seed=42)
    counts = sampled.groupby("label").size().to_dict()
    assert counts[0] == 4
    assert counts[1] == 4
