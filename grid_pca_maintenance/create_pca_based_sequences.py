"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


def create_pca_based_sequences() -> None:
    pca_cols = ["pca_1", "pca_2", "pca_3"]
    X, y = make_pca_lstm_sequences(df, pca_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model_plain = Sequential()
    model_plain.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
    model_plain.add(Dense(1, activation="sigmoid"))
    model_plain.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    model_plain.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
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
    preds_plain = model_plain.predict(X_engine).flatten()
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
