"""Notebook steps (auto-split)."""

import numpy as np


def make_multivariate_lstm_sequences(df, sensors, window=30):
    sequences = []
    labels = []
    grouped = df.groupby("unit")
    for _, group in grouped:
        X = group[sensors].values
        y = group["distress"].values
        for i in range(len(X) - window):
            sequences.append(X[i : i + window])
            labels.append(y[i + window])
    return (np.array(sequences), np.array(labels))
