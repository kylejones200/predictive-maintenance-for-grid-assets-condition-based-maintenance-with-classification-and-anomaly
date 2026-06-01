"""Rolling RMS feature for predictive maintenance signals."""

from __future__ import annotations

import numpy as np


def rolling_rms_features(series: np.ndarray, window: int) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    n = len(s)
    w = max(window, 1)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        start = max(0, i - w + 1)
        sl = s[start : i + 1]
        ms = float((sl * sl).sum()) / len(sl)
        out[i] = ms**0.5
    return out
