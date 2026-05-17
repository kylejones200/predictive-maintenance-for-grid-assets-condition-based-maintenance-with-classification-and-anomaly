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

def make_sure_you_re_only_looking_at_the_sensor_colu() -> None:
    sensor_cols = [col for col in df.columns if col.startswith("sensor")]

    sensor_stats = pd.DataFrame(
        {
            "sensor": sensor_cols,
            "std_dev": [df[col].std() for col in sensor_cols],
            "range": [df[col].max() - df[col].min() for col in sensor_cols],
            "mean": [df[col].mean() for col in sensor_cols],
        }
    )

    sensor_stats = sensor_stats.sort_values(by="std_dev", ascending=False).reset_index(
        drop=True
    )

    print(sensor_stats)

