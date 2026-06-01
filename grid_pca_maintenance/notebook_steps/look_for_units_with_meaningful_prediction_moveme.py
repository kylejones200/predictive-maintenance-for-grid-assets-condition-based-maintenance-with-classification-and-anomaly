"""Notebook steps (auto-split)."""

from tqdm import tqdm


def look_for_units_with_meaningful_prediction_moveme() -> None:
    sensor = "sensor_9"
    window = 30
    unit_ids = df["unit"].unique()
    dynamic_units = []
    for unit_id in tqdm(unit_ids):
        unit_data = df[df["unit"] == unit_id]
        X_unit = make_unit_sequences(unit_data, sensor=sensor, window=window)
        if X_unit.shape[0] == 0:
            continue
        preds = model.predict(X_unit).flatten()
        score = preds.max() - preds.min()
        if score > 0.05:
            dynamic_units.append((unit_id, score, preds))

    dynamic_units.sort(key=lambda x: x[1], reverse=True)
