from typing import Iterable

import numpy as np


def mean_confidence_interval(values: Iterable[float]) -> dict[str, float]:
    arr = np.array(list(values), dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    std = arr.std(ddof=1)
    half_width = 1.96 * std / np.sqrt(len(arr))
    return {
        "mean": mean,
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
    }
