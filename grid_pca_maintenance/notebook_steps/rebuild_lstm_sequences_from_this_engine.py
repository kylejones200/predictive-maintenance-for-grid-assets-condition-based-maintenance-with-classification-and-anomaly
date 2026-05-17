"""Notebook steps (auto-split)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Input, Layer
from tensorflow.keras.models import Model, Sequential
from tqdm import tqdm

def rebuild_lstm_sequences_from_this_engine() -> None:
    unit_id = 3

    unit_data = df[df["unit"] == unit_id].copy()

    X_unit = make_unit_sequences(unit_data)

    preds = model.predict(X_unit)

    cycles = unit_data["time"].values[30:]

    plt.figure(figsize=(10, 4))

    plt.plot(cycles, preds, label="Predicted Distress Probability")

    plt.axhline(0.5, color="red", linestyle="--", label="Threshold")

    plt.title(f"LSTM Distress Prediction for Engine {unit_id}")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.savefig("lstm_distress_over_time.png")

    plt.show()

