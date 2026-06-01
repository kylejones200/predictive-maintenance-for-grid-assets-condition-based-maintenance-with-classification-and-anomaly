"""Notebook steps (auto-split)."""

from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model


def notebook_step_004() -> None:
    input_seq = Input(shape=(X_lstm.shape[1], 1))
    x = LSTM(64, return_sequences=True)(input_seq)
    context, attn_weights = AttentionWithWeights()(x)
    output = Dense(1, activation="sigmoid")(context)
    Model(inputs=input_seq, outputs=[output, attn_weights])
