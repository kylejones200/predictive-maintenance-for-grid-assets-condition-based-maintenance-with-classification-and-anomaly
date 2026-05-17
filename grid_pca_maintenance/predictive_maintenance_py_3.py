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

def predictive_maintenance_py_3() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

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

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(df[selected_sensors])

    pca = PCA(n_components=3)

    pca_factors = pca.fit_transform(X_scaled)

    df[["pca_1", "pca_2", "pca_3"]] = pca_factors

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

    unit_df = df[df["unit"] == 1]

    model = sm.tsa.ARIMA(unit_df["pca_1"], order=(1, 1, 1))

    fit = model.fit()

    forecast = fit.predict(start=1, end=len(unit_df), dynamic=False)

    residuals = unit_df["pca_1"].iloc[1:].values - forecast[1:]

    plt.figure(figsize=(8, 4))

    plt.plot(residuals)

    plt.axhline(y=2 * residuals.std(), color="r", linestyle="--")

    plt.title("ARIMA Residuals on PCA_1")

    plt.savefig("arima_residuals_pca1.png")

    plt.close()

    pca_features = df[["pca_1", "pca_2", "pca_3"]].values

    clf = IsolationForest(contamination=0.05, random_state=42)

    df["anomaly_iforest"] = clf.fit_predict(pca_features) == -1

    plt.figure(figsize=(8, 4))

    plt.hist(df["pca_distance"], bins=50, alpha=0.6, label="PCA Distance")

    plt.axvline(threshold, color="red", linestyle="--", label="Threshold")

    plt.title("Isolation Forest Anomalies")

    plt.legend()

    plt.savefig("iforest_anomaly_pca.png")

    plt.close()

    X_seq, y_seq = make_sequences(df, "sensor_9")

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    rf_clf = RandomForestClassifier(n_estimators=100)

    rf_clf.fit(X_train_rf, y_train_rf)

    y_pred_rf = rf_clf.predict_proba(X_test_rf)[:, 1]

    plt.figure(figsize=(8, 4))

    plt.hist(y_pred_rf, bins=30, alpha=0.7)

    plt.title("Random Forest Distress Probabilities")

    plt.xlabel("Predicted Probability")

    plt.ylabel("Frequency")

    plt.savefig("rf_sequence_classification.png")

    plt.close()

    pca_cols = ["pca_1", "pca_2", "pca_3"]

    X, y = make_pca_lstm_sequences(df, pca_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_lstm = Sequential()

    model_lstm.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))

    model_lstm.add(Dense(1, activation="sigmoid"))

    model_lstm.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    pred_lstm = model_lstm.predict(X_test).flatten()

    plt.figure(figsize=(8, 4))

    plt.plot(pred_lstm, alpha=0.8)

    plt.title("LSTM Distress Predictions")

    plt.xlabel("Sample Index")

    plt.ylabel("Predicted Probability")

    plt.savefig("lstm_predictions.png")

    plt.close()

    input_seq = Input(shape=(X.shape[1], X.shape[2]))

    x = LSTM(64, return_sequences=True)(input_seq)

    context, attn_weights = AttentionWithWeights()(x)

    output = Dense(1, activation="sigmoid", name="pred")(context)

    model_attn = Model(inputs=input_seq, outputs={"pred": output, "attn": attn_weights})

    model_attn.compile(
        optimizer="adam",
        loss={"pred": "binary_crossentropy"},
        metrics={"pred": "accuracy"},
    )

    model_attn.fit(
        X_train, {"pred": y_train}, epochs=10, batch_size=32, validation_split=0.1
    )

    unit_id = 3

    window = 30

    engine_data = df[df["unit"] == unit_id].copy()

    X_engine = np.array(
        [
            engine_data[["pca_1", "pca_2", "pca_3"]].values[i : i + window]
            for i in range(len(engine_data) - window)
        ]
    )

    cycles = engine_data["time"].values[window:]

    pred_lstm_engine = model_lstm.predict(X_engine).flatten()

    pred_dict = model_attn.predict(X_engine)

    pred_attn_engine = pred_dict["pred"].flatten()

    plt.figure(figsize=(10, 4))

    plt.plot(cycles, pred_lstm_engine, label="LSTM")

    plt.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Threshold")

    plt.title(f"LSTM Distress Prediction for Engine {unit_id}")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.tight_layout()

    plt.savefig("lstm_engine_prediction.png")

    plt.close()

    plt.figure(figsize=(10, 4))

    plt.plot(cycles, pred_attn_engine, label="LSTM + Attention")

    plt.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Threshold")

    plt.title(f"LSTM + Attention Prediction for Engine {unit_id}")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.tight_layout()

    plt.savefig("lstm_attention_engine_prediction.png")

    plt.close()

    plt.figure(figsize=(10, 4))

    plt.plot(cycles, pred_lstm_engine, label="LSTM")

    plt.plot(cycles, pred_attn_engine, label="LSTM + Attention", linestyle="--")

    plt.axhline(0.5, color="red", linestyle=":", linewidth=0.8, label="Threshold")

    plt.title(f"Engine {unit_id} – LSTM vs Attention")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.tight_layout()

    plt.savefig("comparison_lstm_vs_attention.png")

    plt.close()

    print("Engine-specific visualizations complete.")

