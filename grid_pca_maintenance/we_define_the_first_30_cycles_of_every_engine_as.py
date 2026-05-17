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

def we_define_the_first_30_cycles_of_every_engine_as() -> None:
    df = pd.read_csv("train_FD001.txt", sep=" ", header=None)

    df.dropna(axis=1, inplace=True)

    columns = ["unit", "time", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)
    ]

    df.columns = columns

    healthy_df = df[df["time"] <= 30]

    features = [col for col in df.columns if col.startswith("sensor")]

    X_healthy = healthy_df[features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_healthy)

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3)

    plt.title("PCA of Healthy Operation")

    plt.xlabel("PC1")

    plt.ylabel("PC2")

    plt.grid(False)

    plt.savefig("pca_healthy.png")

    plt.show()

    X_all_scaled = scaler.transform(df[features])

    X_all_pca = pca.transform(X_all_scaled)

    center = X_pca.mean(axis=0)

    dists = cdist(X_all_pca, [center])

    df["pca_distance"] = dists

    threshold = dists.mean() + 3 * dists.std()

    df["pca_anomaly"] = df["pca_distance"] > threshold

    unit_df = df[df["unit"] == 1]

    sensor = "sensor_9"

    model = sm.tsa.ARIMA(unit_df[sensor], order=(1, 1, 1))

    fit = model.fit()

    forecast = fit.predict(start=1, end=len(unit_df), dynamic=False)

    residuals = unit_df[sensor].iloc[1:].values - forecast[1:]

    plt.plot(residuals)

    plt.axhline(y=2 * residuals.std(), color="r", linestyle="--")

    plt.title(f"Residuals for {sensor}")

    plt.savefig("sensor_residuals.png")

    plt.show()

    X = df[features]

    X_scaled = StandardScaler().fit_transform(X)

    clf = IsolationForest(contamination=0.05, random_state=42)

    df["anomaly_iforest"] = clf.fit_predict(X_scaled)

    df["anomaly_iforest"] = df["anomaly_iforest"] == -1

    rul_df = df.groupby("unit")["time"].max().reset_index()

    rul_df.columns = ["unit", "max_time"]

    df = df.merge(rul_df, on="unit")

    df["RUL"] = df["max_time"] - df["time"]

    df["distress"] = df["RUL"] < 20

    X_seq, y_seq = make_sequences(df, "sensor_9")

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(X_train, y_train)

    print("Accuracy on test set:", clf.score(X_test, y_test))

    X_lstm, y_lstm = make_lstm_sequences(df, "sensor_9")

    X_lstm = X_lstm[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y_lstm, test_size=0.2, random_state=42
    )

    model = Sequential()

    model.add(LSTM(64, input_shape=(X_lstm.shape[1], 1)))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    input_seq = Input(shape=(X_lstm.shape[1], 1))

    x = LSTM(64, return_sequences=True)(input_seq)

    x = Attention()(x)

    output = Dense(1, activation="sigmoid")(x)

    model_attn = Model(inputs=input_seq, outputs=output)

    model_attn.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    model_attn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

