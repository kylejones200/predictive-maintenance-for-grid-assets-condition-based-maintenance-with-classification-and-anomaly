"""Notebook steps (auto-split)."""

import numpy as np


def make_multivariate_sequences(df, sensors, window=30):
    sequences = []
    labels = []
    for unit, group in df.groupby("unit"):
        X = group[sensors].values
        y = group["distress"].values
        for i in range(len(X) - window):
            sequences.append(X[i : i + window])
            labels.append(y[i + window])
    return (np.array(sequences), np.array(labels))
