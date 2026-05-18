"""Notebook steps (auto-split)."""

import matplotlib.pyplot as plt


def rebuild_lstm_sequences_from_this_engine() -> None:
    unit_id = 3
    unit_data = df[df["unit"] == unit_id].copy()
    X_unit = make_unit_sequences(unit_data)
    preds = model.predict(X_unit)
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
