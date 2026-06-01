"""Notebook steps (auto-split)."""

import numpy as np


def make_lstm_sequences(df, sensor, window=30):
    sequences = []
    labels = []
    grouped = df.groupby("unit")
    for _, group in grouped:
        values = group[sensor].values
        targets = group["distress"].values
        for i in range(len(values) - window):
            sequences.append(values[i : i + window])
            labels.append(targets[i + window])
    return (np.array(sequences), np.array(labels))
