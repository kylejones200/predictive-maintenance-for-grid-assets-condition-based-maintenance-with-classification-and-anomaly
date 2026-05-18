"""Auto-split from legacy monolithic script."""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


def notebook_step_014() -> None:
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
    sensor_data = df[selected_sensors]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sensor_data)
    pca = PCA(n_components=3)
    pca_factors = pca.fit_transform(scaled_data)
    df["pca_1"] = pca_factors[:, 0]
    df["pca_2"] = pca_factors[:, 1]
    df["pca_3"] = pca_factors[:, 2]
    pca_columns = ["pca_1", "pca_2", "pca_3"]
    X_pca_seq, y_pca_seq = make_pca_lstm_sequences(df, pca_columns, window=30)
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca_seq, y_pca_seq, test_size=0.2, random_state=42
    )
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    predict_and_plot(model, unit_id=3)
