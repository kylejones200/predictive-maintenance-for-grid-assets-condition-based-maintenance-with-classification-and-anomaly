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

def notebook_step_004() -> None:
    input_seq = Input(shape=(X_lstm.shape[1], 1))

    x = LSTM(64, return_sequences=True)(input_seq)

    context, attn_weights = AttentionWithWeights()(x)

    output = Dense(1, activation="sigmoid")(context)

    model_attn_vis = Model(inputs=input_seq, outputs=[output, attn_weights])

