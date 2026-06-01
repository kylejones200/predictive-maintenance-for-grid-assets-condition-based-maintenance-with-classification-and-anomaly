"""Notebook steps (auto-split)."""

import matplotlib.pyplot as plt


def notebook_step_021() -> None:
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
