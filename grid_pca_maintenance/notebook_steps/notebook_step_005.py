"""Notebook steps (auto-split)."""

import matplotlib.pyplot as plt


def notebook_step_005() -> None:
    sample = X_lstm[0:1]
    pred, attn = model_attn_vis.predict(sample)
    attn = attn[0].squeeze()
    plt.figure(figsize=(10, 3))
    plt.stem(range(len(attn)), attn)
    plt.title("Attention Weights Across Time Steps")
    plt.xlabel("Time Step in Sequence")
    plt.ylabel("Attention Weight")
    plt.savefig("attention_weights.png")
    plt.show()
