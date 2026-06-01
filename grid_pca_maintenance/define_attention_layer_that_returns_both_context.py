"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model


def define_attention_layer_that_returns_both_context() -> None:
    input_seq = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(64, return_sequences=True)(input_seq)
    context, attn_weights = AttentionWithWeights()(x)
    output = Dense(1, activation="sigmoid")(context)
    model_attn_vis = Model(inputs=input_seq, outputs=[output, attn_weights])
    model_attn_vis.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    model_attn_vis.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    unit_id = 3
    unit_data = df[df["unit"] == unit_id].copy()
    pca_cols = ["pca_1", "pca_2", "pca_3"]
    values = unit_data[pca_cols].values
    sequences = [values[i : i + 30] for i in range(len(values) - 30)]
    X_unit = np.array(sequences)
    sample = X_unit[0:1]
    pred, attn = model_attn_vis.predict(sample)
    attn = attn[0].squeeze()
    plt.figure(figsize=(10, 3))
    plt.stem(range(len(attn)), attn, use_line_collection=True)
    plt.title("Attention Weights Across Time Steps")
    plt.xlabel("Time Step in Sequence")
    plt.ylabel("Attention Weight")
    plt.grid(True)
    plt.savefig("attention_weights_pca.png")
    plt.show()
