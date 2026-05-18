"""Notebook steps (auto-split)."""

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


def select_a_few_high_value_sensors() -> None:
    selected_sensors = ["sensor_2", "sensor_3", "sensor_14", "sensor_15"]
    X_seq, y_seq = make_multivariate_sequences(df, selected_sensors)
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
