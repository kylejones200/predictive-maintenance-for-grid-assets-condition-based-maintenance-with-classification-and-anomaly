"""Notebook steps (auto-split)."""

import matplotlib.pyplot as plt
import numpy as np


def predict_and_plot(model, unit_id=3, window=30):
    unit_data = df[df["unit"] == unit_id].copy()
    values = unit_data[pca_columns].values
    sequences = [values[i : i + window] for i in range(len(values) - window)]
    X_unit = np.array(sequences)
    preds = model.predict(X_unit).flatten()
    cycles = unit_data["time"].values[window:]
    plt.figure(figsize=(10, 4))
    plt.plot(cycles, preds, label="Predicted Distress Probability")
    plt.axhline(0.5, color="red", linestyle="--", label="Threshold")
    plt.title(f"LSTM Prediction on PCA Features — Engine {unit_id}")
    plt.xlabel("Cycle")
    plt.ylabel("Distress Probability")
    plt.legend()
    plt.grid(True)
    plt.show()
