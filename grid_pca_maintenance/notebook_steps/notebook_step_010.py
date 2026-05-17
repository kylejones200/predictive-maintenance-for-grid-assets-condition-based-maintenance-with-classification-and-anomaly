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

def notebook_step_010() -> None:
    top_unit_id = dynamic_units[0][0]

    top_preds = dynamic_units[0][2]

    unit_data = df[df["unit"] == top_unit_id].copy()

    cycles = unit_data["time"].values[window:]

    plt.figure(figsize=(10, 4))

    plt.plot(cycles, top_preds, label="Predicted Distress Probability")

    plt.axhline(0.5, color="red", linestyle="--", label="Threshold")

    plt.title(f"LSTM Distress Prediction for Engine {top_unit_id}")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.savefig(f"lstm_unit_{top_unit_id}_dynamic.png")

    plt.show()

