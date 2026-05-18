"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt


def lstm() -> None:
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
