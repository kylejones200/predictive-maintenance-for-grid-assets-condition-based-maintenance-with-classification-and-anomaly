"""Generated from Jupyter notebook: PCA mahalanoibis

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    df = pd.read_csv("train_FD001.txt", sep="\\s+", header=None)
    df.dropna(axis=1, inplace=True)
    df.columns = ["unit", "time", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)
    ]
    selected_sensors = [
        "sensor_9",
        "sensor_14",
        "sensor_4",
        "sensor_3",
        "sensor_17",
        "sensor_2",
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[selected_sensors])
    pca = PCA(n_components=3)
    pca_factors = pca.fit_transform(X_scaled)
    df[["pca_1", "pca_2", "pca_3"]] = pca_factors
    df["distance"] = np.sqrt(df["pca_1"] ** 2 + df["pca_2"] ** 2 + df["pca_3"] ** 2)
    plt.figure(figsize=(8, 4))
    plt.plot(df[df["unit"] == 3]["distance"], alpha=0.8)
    plt.title("Health Index for Engine 3 using Mahalanobis Distance")
    plt.xlabel("Cycle")
    plt.ylabel("Health Index")
    plt.axhline(4, color="red", linestyle="--", linewidth=0.8, label="Threshold")
    plt.savefig("health_index.png")
    plt.show()
    count = df[df["unit"] == 78]["distance"] > 4
    sum(count)
    df["unit"].nunique()
    plt.figure(figsize=(8, 4))
    plt.plot(df[df["unit"] == 72]["time"], df[df["unit"] == 72]["distance"], alpha=0.8)
    plt.title("Health Index for Engine 72 using Mahalanobis Distance")
    plt.xlabel("Cycle")
    plt.ylabel("Health Index")
    plt.axhline(4, color="red", linestyle="--", linewidth=0.8, label="Threshold")
    plt.savefig("health_index.png")
    plt.show()
    df.head()
    df[df["unit"] == 78]["distance"].tail(50)


def main() -> None:
    main()


if __name__ == "__main__":
    main()
