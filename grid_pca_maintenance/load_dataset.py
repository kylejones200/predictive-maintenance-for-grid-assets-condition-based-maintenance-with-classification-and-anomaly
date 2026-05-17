"""Auto-split from legacy monolithic script."""

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

def load_dataset() -> None:
    df = pd.read_csv("train_FD001.txt", sep="\\s+", header=None)

    df.dropna(axis=1, inplace=True)

    df.columns = ["unit", "time", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)
    ]

    rul = pd.read_csv("RUL_FD001.txt", header=None)

    rul.columns = ["max_RUL"]

    rul["unit"] = rul.index + 1

    last_cycle = df.groupby("unit")["time"].max().reset_index()

    last_cycle.columns = ["unit", "max_time"]

    df = df.merge(last_cycle, on="unit")

    df["RUL"] = df["max_time"] - df["time"]

    df["distress"] = df["RUL"] < 20

    selected_sensors = [
        "sensor_9",
        "sensor_14",
        "sensor_4",
        "sensor_3",
        "sensor_17",
        "sensor_2",
    ]

    X_seq, y_seq = make_multivariate_lstm_sequences(df, selected_sensors, window=30)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    model = Sequential()

    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    input_seq = Input(shape=(X_train.shape[1], X_train.shape[2]))

    x = LSTM(64, return_sequences=True)(input_seq)

    x = Attention()(x)

    output = Dense(1, activation="sigmoid")(x)

    model_attn = Model(inputs=input_seq, outputs=output)

    model_attn.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    model_attn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    predict_and_plot(model_attn, unit_id=3)

