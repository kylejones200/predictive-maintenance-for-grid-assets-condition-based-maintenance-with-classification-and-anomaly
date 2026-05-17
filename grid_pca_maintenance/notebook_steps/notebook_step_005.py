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

def notebook_step_005() -> None:
    sample = X_lstm[0:1]

    pred, attn = model_attn_vis.predict(sample)

    attn = attn[0].squeeze()

    plt.figure(figsize=(10, 3))

    plt.stem(range(len(attn)), attn)

    plt.title("Attention Weights Across Time Steps")

    plt.xlabel("Time Step in Sequence")

    plt.ylabel("Attention Weight")

    plt.savefig("attention_weights.png")

    plt.show()

