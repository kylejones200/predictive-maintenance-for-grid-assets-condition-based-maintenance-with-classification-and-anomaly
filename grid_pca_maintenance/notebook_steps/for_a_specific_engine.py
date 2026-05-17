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

def for_a_specific_engine() -> None:
    unit_id = 3

    unit_data = df[df["unit"] == unit_id].copy()

    X_unit = np.array(
        [unit_data[pca_cols].values[i : i + 30] for i in range(len(unit_data) - 30)]
    )

    cycles = unit_data["time"].values[30:]

    preds_plain = model_lstm.predict(X_unit).flatten()

    preds_dict = model_attn.predict(X_unit)

    preds_attn = preds_dict["pred"].flatten()

