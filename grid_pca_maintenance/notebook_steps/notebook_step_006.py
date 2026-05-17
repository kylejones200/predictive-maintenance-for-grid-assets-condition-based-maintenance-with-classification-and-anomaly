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

def notebook_step_006() -> None:
    interesting_units = []

    sensor = "sensor_15"

    window = 30

    unit_ids = df["unit"].unique()

    for unit_id in tqdm(unit_ids):
        unit_data = df[df["unit"] == unit_id]
        if len(unit_data) <= window:
            continue
        X_unit = make_unit_sequences(unit_data, sensor=sensor, window=window)
        preds = model.predict(X_unit).flatten()
        if preds.size == 0:
            continue
        score = preds.max() - preds.min()
        interesting_units.append((unit_id, score, preds.mean(), preds.std()))

