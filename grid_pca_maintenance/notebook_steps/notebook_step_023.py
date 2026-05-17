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

def notebook_step_023() -> None:
    healthy_df = df[df["time"] <= 30]

    X_pca = healthy_df[["pca_1", "pca_2"]].values

    center = X_pca.mean(axis=0)

    df["pca_distance"] = np.linalg.norm(df[["pca_1", "pca_2"]].values - center, axis=1)

    threshold = df["pca_distance"].mean() + 3 * df["pca_distance"].std()

    df["pca_anomaly"] = df["pca_distance"] > threshold

    plt.figure(figsize=(8, 4))

    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3)

    plt.title("PCA of Healthy Operation")

    plt.xlabel("PC1")

    plt.ylabel("PC2")

    plt.grid(False)

    plt.savefig("pca_healthy_bounds.png")

    plt.close()

