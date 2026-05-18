"""Notebook steps (auto-split)."""

import matplotlib.pyplot as plt


def notebook_step_008() -> None:
    unit_id = top_units[0][0]
    unit_data = df[df["unit"] == unit_id].copy()
    X_unit = make_unit_sequences(unit_data, sensor=sensor, window=30)
    preds = model.predict(X_unit).flatten()
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
