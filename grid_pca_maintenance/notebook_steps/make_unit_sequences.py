"""Notebook steps (auto-split)."""

import numpy as np


def make_unit_sequences(unit_data, sensor, window=30):
    values = unit_data[sensor].values
    if len(values) <= window:
        return np.array([])
    sequences = [values[i : i + window] for i in range(len(values) - window)]
    return np.array(sequences)[..., np.newaxis]
