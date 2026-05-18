"""Notebook steps (auto-split)."""

import pandas as pd


def make_sure_you_re_only_looking_at_the_sensor_colu() -> None:
    sensor_cols = [col for col in df.columns if col.startswith("sensor")]
    sensor_stats = pd.DataFrame(
        {
            "sensor": sensor_cols,
            "std_dev": [df[col].std() for col in sensor_cols],
            "range": [df[col].max() - df[col].min() for col in sensor_cols],
            "mean": [df[col].mean() for col in sensor_cols],
        }
    )
    sensor_stats = sensor_stats.sort_values(by="std_dev", ascending=False).reset_index(
        drop=True
    )
    print(sensor_stats)
