"""Generated from Jupyter notebook: predictive maintenance with PCA

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Attention(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1), initializer="random_normal"
        )

    def call(self, inputs):
        scores = torch.matmul(inputs, self.W)
        weights = torch.softmax(scores, axis=1, dim=-1)
        context = inputs * weights.sum(dim=1)
        return context


class AttentionWithWeights(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1), initializer="random_normal"
        )

    def call(self, inputs):
        scores = torch.matmul(inputs, self.W)
        weights = torch.softmax(scores, axis=1, dim=-1)
        output = inputs * weights.sum(dim=1)
        return (output, weights)


class _LSTMForecaster(nn.Module):
    """LSTM forecaster (auto-generated PyTorch replacement for Keras Sequential)."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        output_size: int = 1,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


def _predict_torch(model: nn.Module, X_test) -> "np.ndarray":
    """Replace model.predict()."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()


def _train_torch(
    model: nn.Module,
    X_train,
    y_train,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    validation_split: float = 0.2,
    patience: int = 15,
) -> nn.Module:
    """Standard training loop replacing  + model.fit()."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    n_val = max(1, int(len(X_t) * validation_split))
    X_val, y_val = (X_t[-n_val:], y_t[-n_val:])
    X_tr, y_tr = (X_t[:-n_val], y_t[:-n_val])
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best, wait = (float("inf"), 0)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best:
            best, wait = (val_loss, 0)
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def make_lstm_sequences(df, sensor, window=30):
    sequences = []
    labels = []
    grouped = df.groupby("unit")
    for _, group in grouped:
        values = group[sensor].values
        targets = group["distress"].values
        for i in range(len(values) - window):
            sequences.append(values[i : i + window])
            labels.append(targets[i + window])
    return (np.array(sequences), np.array(labels))


def make_multivariate_lstm_sequences(df, sensors, window=30):
    sequences = []
    labels = []
    grouped = df.groupby("unit")
    for _, group in grouped:
        X = group[sensors].values
        y = group["distress"].values
        for i in range(len(X) - window):
            sequences.append(X[i : i + window])
            labels.append(y[i + window])
    return (np.array(sequences), np.array(labels))


def make_multivariate_sequences(df, sensors, window=30):
    sequences = []
    labels = []
    for unit, group in df.groupby("unit"):
        X = group[sensors].values
        y = group["distress"].values
        for i in range(len(X) - window):
            sequences.append(X[i : i + window])
            labels.append(y[i + window])
    return (np.array(sequences), np.array(labels))


def make_pca_lstm_sequences(df, pca_cols, window=30):
    sequences, labels = ([], [])
    for _, group in df.groupby("unit"):
        values = group[pca_cols].values
        targets = group["distress"].values
        for i in range(len(values) - window):
            sequences.append(values[i : i + window])
            labels.append(targets[i + window])
    return (np.array(sequences), np.array(labels))


def make_sequences(df, sensor, window=30):
    sequences, labels = ([], [])
    for _, group in df.groupby("unit"):
        values = group[sensor].values
        targets = group["distress"].values
        for i in range(len(values) - window):
            sequences.append(values[i : i + window])
            labels.append(targets[i + window])
    return (np.array(sequences), np.array(labels))


def make_unit_sequences(unit_data, sensor, window=30):
    values = unit_data[sensor].values
    if len(values) <= window:
        return np.array([])
    sequences = [values[i : i + window] for i in range(len(values) - window)]
    return np.array(sequences)[..., np.newaxis]


def predict_and_plot(model, unit_id=3, window=30):
    unit_data = df[df["unit"] == unit_id].copy()
    values = unit_data[pca_columns].values
    sequences = [values[i : i + window] for i in range(len(values) - window)]
    X_unit = np.array(sequences)
    preds = _predict_torch(model, X_unit).flatten()
    cycles = unit_data["time"].values[window:]
    plt.figure(figsize=(10, 4))
    plt.plot(cycles, preds, label="Predicted Distress Probability")
    plt.axhline(0.5, color="red", linestyle="--", label="Threshold")
    plt.title(f"LSTM Prediction on PCA Features — Engine {unit_id}")
    plt.xlabel("Cycle")
    plt.ylabel("Distress Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
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

    forecast = _predict_torch(fit, start=1, end=len(unit_df), dynamic=False)

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

    _train_torch(clf, X_train, y_train)

    print("Accuracy on test set:", clf.score(X_test, y_test))

    X_lstm, y_lstm = make_lstm_sequences(df, "sensor_9")

    X_lstm = X_lstm[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y_lstm, test_size=0.2, random_state=42
    )

    model = Sequential()

    model.add(LSTM(64, input_shape=(X_lstm.shape[1], 1)))

    model.add(Dense(1, activation="sigmoid"))

    _train_torch(model, X_train, y_train)

    input_seq = Input(shape=(X_lstm.shape[1], 1))

    x = LSTM(64, return_sequences=True)(input_seq)

    x = Attention()(x)

    output = Dense(1, activation="sigmoid")(x)

    model_attn = Model(inputs=input_seq, outputs=output)

    _train_torch(model_attn, X_train, y_train)

    unit_id = 3

    unit_data = df[df["unit"] == unit_id].copy()

    X_unit = make_unit_sequences(unit_data)

    preds = _predict_torch(model, X_unit)

    cycles = unit_data["time"].values[30:]

    plt.figure(figsize=(10, 4))

    plt.plot(cycles, preds, label="Predicted Distress Probability")

    plt.axhline(0.5, color="red", linestyle="--", label="Threshold")

    plt.title(f"LSTM Distress Prediction for Engine {unit_id}")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.savefig("lstm_distress_over_time.png")

    plt.show()

    input_seq = Input(shape=(X_lstm.shape[1], 1))

    x = LSTM(64, return_sequences=True)(input_seq)

    context, attn_weights = AttentionWithWeights()(x)

    output = Dense(1, activation="sigmoid")(context)

    model_attn_vis = Model(inputs=input_seq, outputs=[output, attn_weights])

    sample = X_lstm[0:1]

    pred, attn = _predict_torch(model_attn_vis, sample)

    attn = attn[0].squeeze()

    plt.figure(figsize=(10, 3))

    plt.stem(range(len(attn)), attn)

    plt.title("Attention Weights Across Time Steps")

    plt.xlabel("Time Step in Sequence")

    plt.ylabel("Attention Weight")

    plt.savefig("attention_weights.png")

    plt.show()

    interesting_units = []

    sensor = "sensor_15"

    window = 30

    unit_ids = df["unit"].unique()

    for unit_id in tqdm(unit_ids):
        unit_data = df[df["unit"] == unit_id]
        if len(unit_data) <= window:
            continue
        X_unit = make_unit_sequences(unit_data, sensor=sensor, window=window)
        preds = _predict_torch(model, X_unit).flatten()
        if preds.size == 0:
            continue
        score = preds.max() - preds.min()
        interesting_units.append((unit_id, score, preds.mean(), preds.std()))

    interesting_units.sort(key=lambda x: x[1], reverse=True)

    top_units = interesting_units[:5]

    for unit_id, score, mean, std in top_units:
        print(f"Unit {unit_id} | Δ = {score:.3f} | μ = {mean:.3f} | σ = {std:.3f}")

    unit_id = top_units[0][0]

    unit_data = df[df["unit"] == unit_id].copy()

    X_unit = make_unit_sequences(unit_data, sensor=sensor, window=30)

    preds = _predict_torch(model, X_unit).flatten()

    cycles = unit_data["time"].values[30:]

    plt.figure(figsize=(10, 4))

    plt.plot(cycles, preds, label="Predicted Distress Probability")

    plt.axhline(0.5, color="red", linestyle="--", label="Threshold")

    plt.title(f"LSTM Distress Prediction for Engine {unit_id}")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.savefig(f"lstm_unit_{unit_id}_interesting.png")

    plt.show()

    sensor = "sensor_9"

    window = 30

    unit_ids = df["unit"].unique()

    dynamic_units = []

    for unit_id in tqdm(unit_ids):
        unit_data = df[df["unit"] == unit_id]
        X_unit = make_unit_sequences(unit_data, sensor=sensor, window=window)
        if X_unit.shape[0] == 0:
            continue
        preds = _predict_torch(model, X_unit).flatten()
        score = preds.max() - preds.min()
        if score > 0.05:
            dynamic_units.append((unit_id, score, preds))

    dynamic_units.sort(key=lambda x: x[1], reverse=True)

    top_unit_id = dynamic_units[0][0]

    top_preds = dynamic_units[0][2]

    unit_data = df[df["unit"] == top_unit_id].copy()

    cycles = unit_data["time"].values[window:]

    plt.figure(figsize=(10, 4))

    plt.plot(cycles, top_preds, label="Predicted Distress Probability")

    plt.axhline(0.5, color="red", linestyle="--", label="Threshold")

    plt.title(f"LSTM Distress Prediction for Engine {top_unit_id}")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.savefig(f"lstm_unit_{top_unit_id}_dynamic.png")

    plt.show()

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

    selected_sensors = ["sensor_2", "sensor_3", "sensor_14", "sensor_15"]

    X_seq, y_seq = make_multivariate_sequences(df, selected_sensors)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    model = Sequential()

    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))

    model.add(Dense(1, activation="sigmoid"))

    _train_torch(model, X_train, y_train)

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

    _train_torch(model, X_train, y_train)

    input_seq = Input(shape=(X_train.shape[1], X_train.shape[2]))

    x = LSTM(64, return_sequences=True)(input_seq)

    x = Attention()(x)

    output = Dense(1, activation="sigmoid")(x)

    model_attn = Model(inputs=input_seq, outputs=output)

    _train_torch(model_attn, X_train, y_train)

    predict_and_plot(model_attn, unit_id=3)

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

    _train_torch(model, X_train, y_train)

    predict_and_plot(model, unit_id=3)

    input_seq = Input(shape=(X_train.shape[1], X_train.shape[2]))

    x = LSTM(64, return_sequences=True)(input_seq)

    context, attn_weights = AttentionWithWeights()(x)

    output = Dense(1, activation="sigmoid")(context)

    model_attn_vis = Model(inputs=input_seq, outputs=[output, attn_weights])

    _train_torch(model_attn_vis, X_train, y_train)

    unit_id = 3

    unit_data = df[df["unit"] == unit_id].copy()

    pca_cols = ["pca_1", "pca_2", "pca_3"]

    values = unit_data[pca_cols].values

    sequences = [values[i : i + 30] for i in range(len(values) - 30)]

    X_unit = np.array(sequences)

    sample = X_unit[0:1]

    pred, attn = _predict_torch(model_attn_vis, sample)

    attn = attn[0].squeeze()

    plt.figure(figsize=(10, 3))

    plt.stem(range(len(attn)), attn, use_line_collection=True)

    plt.title("Attention Weights Across Time Steps")

    plt.xlabel("Time Step in Sequence")

    plt.ylabel("Attention Weight")

    plt.grid(True)

    plt.savefig("attention_weights_pca.png")

    plt.show()

    pca_cols = ["pca_1", "pca_2", "pca_3"]

    X, y = make_pca_lstm_sequences(df, pca_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_plain = Sequential()

    model_plain.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))

    model_plain.add(Dense(1, activation="sigmoid"))

    _train_torch(model_plain, X_train, y_train)

    unit_id = 3

    window = 30

    engine_data = df[df["unit"] == unit_id].copy()

    X_engine = np.array(
        [
            engine_data[pca_cols].values[i : i + window]
            for i in range(len(engine_data) - window)
        ]
    )

    cycles = engine_data["time"].values[window:]

    preds_plain = _predict_torch(model_plain, X_engine).flatten()

    plt.figure(figsize=(10, 4))

    plt.plot(cycles, preds_plain, label="LSTM (No Attention)")

    plt.axhline(0.5, color="red", linestyle="--", label="Threshold")

    plt.title(f"LSTM Prediction Without Attention — Engine {unit_id}")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("distress_prediction_no_attention.png")

    plt.show()

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

    plt.grid(True)

    plt.savefig("pca_healthy_bounds.png")

    plt.close()

    unit_df = df[df["unit"] == 1]

    model = sm.tsa.ARIMA(unit_df["pca_1"], order=(1, 1, 1))

    fit = model.fit()

    forecast = _predict_torch(fit, start=1, end=len(unit_df), dynamic=False)

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

    _train_torch(rf_clf, X_train_rf, y_train_rf)

    pca_cols = ["pca_1", "pca_2", "pca_3"]

    X, y = make_pca_lstm_sequences(df, pca_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_lstm = Sequential()

    model_lstm.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))

    model_lstm.add(Dense(1, activation="sigmoid"))

    _train_torch(model_lstm, X_train, y_train)

    input_seq = Input(shape=(X.shape[1], X.shape[2]))

    x = LSTM(64, return_sequences=True)(input_seq)

    context, attn_weights = AttentionWithWeights()(x)

    output = Dense(1, activation="sigmoid")(context)

    model_attn = Model(inputs=input_seq, outputs=[output, attn_weights])

    _train_torch(model_attn, X_train, y_train)

    print("Training complete. All models and visualizations are ready.")

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

    plt.grid(True)

    plt.savefig("pca_healthy_bounds.png")

    plt.close()

    unit_df = df[df["unit"] == 1]

    model = sm.tsa.ARIMA(unit_df["pca_1"], order=(1, 1, 1))

    fit = model.fit()

    forecast = _predict_torch(fit, start=1, end=len(unit_df), dynamic=False)

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

    _train_torch(rf_clf, X_train_rf, y_train_rf)

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

    _train_torch(model_lstm, X_train, y_train)

    pred_lstm = _predict_torch(model_lstm, X_test).flatten()

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

    _train_torch(model_attn, X_train, {"pred": y_train})

    pred_dict = _predict_torch(model_attn, X_test)

    pred_attn = pred_dict["pred"]

    attn = pred_dict["attn"]

    plt.figure(figsize=(8, 4))

    plt.plot(pred_attn.flatten(), alpha=0.8)

    plt.title("LSTM + Attention Distress Predictions")

    plt.xlabel("Sample Index")

    plt.ylabel("Predicted Probability")

    plt.savefig("lstm_attention_predictions.png")

    plt.close()

    print("Training complete. All models and visualizations are ready.")

    unit_id = 3

    unit_data = df[df["unit"] == unit_id].copy()

    X_unit = np.array(
        [unit_data[pca_cols].values[i : i + 30] for i in range(len(unit_data) - 30)]
    )

    cycles = unit_data["time"].values[30:]

    preds_plain = _predict_torch(model_lstm, X_unit).flatten()

    preds_dict = _predict_torch(model_attn, X_unit)

    preds_attn = preds_dict["pred"].flatten()

    plt.figure(figsize=(10, 4))

    plt.plot(cycles, preds_plain, label="LSTM")

    plt.axhline(0.5, color="red", linestyle="--", label="Threshold")

    plt.title(f"LSTM Distress Prediction for Engine {unit_id}")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("lstm_engine_prediction.png")

    plt.show()

    plt.figure(figsize=(10, 4))

    plt.plot(cycles, preds_attn, label="LSTM + Attention")

    plt.axhline(0.5, color="red", linestyle="--", label="Threshold")

    plt.title(f"LSTM + Attention Prediction for Engine {unit_id}")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("lstm_attention_engine_prediction.png")

    plt.show()

    plt.figure(figsize=(10, 4))

    plt.plot(cycles, preds_plain, label="LSTM")

    plt.plot(cycles, preds_attn, label="LSTM + Attention", linestyle="--")

    plt.axhline(0.5, color="red", linestyle=":", label="Threshold")

    plt.title(f"Engine {unit_id} – LSTM vs Attention")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("comparison_lstm_vs_attention.png")

    plt.show()

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

    forecast = _predict_torch(fit, start=1, end=len(unit_df), dynamic=False)

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

    _train_torch(rf_clf, X_train_rf, y_train_rf)

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

    _train_torch(model_lstm, X_train, y_train)

    pred_lstm = _predict_torch(model_lstm, X_test).flatten()

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

    _train_torch(model_attn, X_train, {"pred": y_train})

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

    pred_lstm_engine = _predict_torch(model_lstm, X_engine).flatten()

    pred_dict = _predict_torch(model_attn, X_engine)

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

    input_seq = Input(shape=(X_lstm.shape[1], 1))

    x = LSTM(64, return_sequences=True)(input_seq)

    context, attn_weights = AttentionWithWeights()(x)

    output = Dense(1, activation="sigmoid")(context)

    model_attn_vis = Model(inputs=input_seq, outputs=[output, attn_weights])

    sample = X_lstm[0:1]

    pred, attn = _predict_torch(model_attn_vis, sample)

    attn = attn[0].squeeze()

    plt.figure(figsize=(10, 3))

    plt.stem(range(len(attn)), attn)

    plt.title("Attention Weights Across Time Steps")

    plt.xlabel("Time Step in Sequence")

    plt.ylabel("Attention Weight")

    plt.savefig("attention_weights.png")

    plt.show()

    prompt = "\nYou are looking at a time series forecasting model for sensor_9 from a jet engine.\nThe model starts to deviate significantly from actual values after cycle 150.\nWhat might cause this, and what could you try to improve it?\n"

    response = openai.ChatCompletion.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )

    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
