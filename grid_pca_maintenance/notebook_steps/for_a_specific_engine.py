"""Notebook steps (auto-split)."""

import numpy as np


def for_a_specific_engine() -> None:
    unit_id = 3
    unit_data = df[df["unit"] == unit_id].copy()
    X_unit = np.array(
        [unit_data[pca_cols].values[i : i + 30] for i in range(len(unit_data) - 30)]
    )
    unit_data["time"].values[30:]
    model_lstm.predict(X_unit).flatten()
    preds_dict = model_attn.predict(X_unit)
    preds_dict["pred"].flatten()
