"""Notebook steps (auto-split)."""

import numpy as np


def make_sequences(df, sensor, window=30):
    sequences, labels = ([], [])
    for _, group in df.groupby("unit"):
        values = group[sensor].values
        targets = group["distress"].values
        for i in range(len(values) - window):
            sequences.append(values[i : i + window])
            labels.append(targets[i + window])
    return (np.array(sequences), np.array(labels))
