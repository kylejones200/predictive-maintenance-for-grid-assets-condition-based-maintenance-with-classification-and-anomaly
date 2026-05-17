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

def look_for_units_with_meaningful_prediction_moveme() -> None:
    sensor = "sensor_9"

    window = 30

    unit_ids = df["unit"].unique()

    dynamic_units = []

    for unit_id in tqdm(unit_ids):
        unit_data = df[df["unit"] == unit_id]
        X_unit = make_unit_sequences(unit_data, sensor=sensor, window=window)
        if X_unit.shape[0] == 0:
            continue
        preds = model.predict(X_unit).flatten()
        score = preds.max() - preds.min()
        if score > 0.05:
            dynamic_units.append((unit_id, score, preds))

    dynamic_units.sort(key=lambda x: x[1], reverse=True)

