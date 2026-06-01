"""Notebook steps (auto-split)."""

from tqdm import tqdm


def notebook_step_006() -> None:
    interesting_units = []
    sensor = "sensor_15"
    window = 30
    unit_ids = df["unit"].unique()
    for unit_id in tqdm(unit_ids):
        unit_data = df[df["unit"] == unit_id]
        if len(unit_data) <= window:
            continue
        X_unit = make_unit_sequences(unit_data, sensor=sensor, window=window)
        preds = model.predict(X_unit).flatten()
        if preds.size == 0:
            continue
        score = preds.max() - preds.min()
        interesting_units.append((unit_id, score, preds.mean(), preds.std()))
