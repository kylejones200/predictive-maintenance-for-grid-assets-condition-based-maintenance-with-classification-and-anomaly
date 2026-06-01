"""Notebook steps (auto-split)."""

import matplotlib.pyplot as plt


def notebook_step_010() -> None:
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
