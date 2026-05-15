"""PCA + Mahalanobis distance and a Keras autoencoder for multivariate anomaly detection.

Converted from `PCA anomaly.ipynb`: grid asset sensor CSV, train/test split, scaling,
2D PCA with distance-based flags, then dense autoencoder reconstruction error.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import preprocessing
from sklearn.decomposition import PCA
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense

# -----------------------------------------------------------------------------
# Mahalanobis helpers (PCA space)
# -----------------------------------------------------------------------------


def _is_positive_definite(a: np.ndarray) -> bool:
    if not np.allclose(a, a.T):
        return False
    try:
        np.linalg.cholesky(a)
        return True
    except np.linalg.LinAlgError:
        return False


def compute_covariance_matrices(data: np.ndarray, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    covariance_matrix = np.cov(data, rowvar=False)
    if not _is_positive_definite(covariance_matrix):
        raise RuntimeError("Covariance matrix is not positive definite.")
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    if not _is_positive_definite(inv_covariance_matrix):
        raise RuntimeError("Inverse of covariance matrix is not positive definite.")
    return covariance_matrix, inv_covariance_matrix


def mahalanobis_distances(inv_cov_matrix: np.ndarray, mean_distr: np.ndarray, data: np.ndarray) -> np.ndarray:
    diff = data - mean_distr
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov_matrix, diff))


def md_threshold(dist: np.ndarray, extreme: bool = False) -> float:
    k = 3.0 if extreme else 2.0
    return float(np.mean(dist) * k)


# -----------------------------------------------------------------------------
# Data & pipelines
# -----------------------------------------------------------------------------


def load_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date-Time"] = df["Date-Time"].astype("datetime64[ns]")
    df.drop("health", axis=1, inplace=True, errors="ignore")
    df.set_index("Date-Time", inplace=True)
    return df


def main() -> None:
    sns.set(color_codes=True)
    root = Path(__file__).resolve().parent
    csv_path = root / "ranfle.csv"

    np.random.seed(10)
    tf.random.set_seed(10)

    df = load_frame(csv_path)
    print(df.head())

    cut_point = "2018-06-11 20:18:00"
    dataset_train = df.loc[:cut_point]
    dataset_test = df.loc[cut_point:]
    dataset_train.plot(figsize=(12, 6))
    plt.title("Train segment")
    plt.tight_layout()
    plt.show()

    scaler = preprocessing.MinMaxScaler()
    x_train = pd.DataFrame(
        scaler.fit_transform(dataset_train),
        columns=dataset_train.columns,
        index=dataset_train.index,
    )
    x_test = pd.DataFrame(
        scaler.transform(dataset_test),
        columns=dataset_test.columns,
        index=dataset_test.index,
    )

    # --- PCA + Mahalanobis ---
    pca = PCA(n_components=2, svd_solver="full")
    x_train_pca = pd.DataFrame(pca.fit_transform(x_train), index=x_train.index)
    x_test_pca = pd.DataFrame(pca.transform(x_test), index=x_test.index)

    data_train = np.asarray(x_train_pca.values, dtype=float)
    data_test = np.asarray(x_test_pca.values, dtype=float)

    _, inv_cov_matrix = compute_covariance_matrices(data_train)
    mean_distr = data_train.mean(axis=0)

    dist_test = mahalanobis_distances(inv_cov_matrix, mean_distr, data_test)
    dist_train = mahalanobis_distances(inv_cov_matrix, mean_distr, data_train)
    threshold = md_threshold(dist_train, extreme=True)

    plt.figure()
    sns.histplot(np.square(dist_train), bins=10, kde=False)
    plt.xlim(0.0, 15)
    plt.title("Squared Mahalanobis distance (train)")
    plt.tight_layout()
    plt.show()

    plt.figure()
    sns.histplot(dist_train, bins=10, kde=True, color="green")
    plt.xlim(0.0, 5)
    plt.xlabel("Mahalanobis distance")
    plt.tight_layout()
    plt.show()

    anomaly_train = pd.DataFrame(
        {"Mob dist": dist_train, "Thresh": threshold, "Anomaly": dist_train > threshold},
        index=x_train_pca.index,
    )
    anomaly_test = pd.DataFrame(
        {"Mob dist": dist_test, "Thresh": threshold, "Anomaly": dist_test > threshold},
        index=x_test_pca.index,
    )
    print(anomaly_test.head())

    anomaly_alldata = pd.concat([anomaly_train, anomaly_test])
    anomaly_alldata.to_csv(root / "Anomaly_distance.csv")
    anomaly_alldata.plot(logy=True, figsize=(10, 6), ylim=[1e-1, 1e3], color=["green", "red"])
    plt.title("Mahalanobis anomaly scores")
    plt.tight_layout()
    plt.show()

    # --- Autoencoder on scaled features ---
    act_func = "elu"
    n_features = x_train.shape[1]
    model = Sequential(
        [
            Dense(
                10,
                activation=act_func,
                kernel_initializer="glorot_uniform",
                kernel_regularizer=regularizers.l2(0.0),
                input_shape=(n_features,),
            ),
            Dense(2, activation=act_func, kernel_initializer="glorot_uniform"),
            Dense(10, activation=act_func, kernel_initializer="glorot_uniform"),
            Dense(n_features, kernel_initializer="glorot_uniform"),
        ]
    )
    model.compile(loss="mse", optimizer="adam")

    num_epochs = 100
    batch_size = 10
    history = model.fit(
        np.asarray(x_train),
        np.asarray(x_train),
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.05,
        verbose=1,
    )

    plt.plot(history.history["loss"], "b", label="Training loss")
    plt.plot(history.history["val_loss"], "r", label="Validation loss")
    plt.legend(loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.ylim(0, 0.02)
    plt.tight_layout()
    plt.show()

    x_pred_train = pd.DataFrame(model.predict(np.asarray(x_train), verbose=0), columns=x_train.columns, index=x_train.index)
    scored_train = pd.DataFrame(index=x_train.index)
    scored_train["Loss_mae"] = np.mean(np.abs(x_pred_train - x_train), axis=1)

    plt.figure()
    sns.histplot(scored_train["Loss_mae"], bins=10, kde=True, color="blue")
    plt.xlim(0.0, 0.5)
    plt.title("Train reconstruction MAE")
    plt.tight_layout()
    plt.show()

    x_pred_test = pd.DataFrame(model.predict(np.asarray(x_test), verbose=0), columns=x_test.columns, index=x_test.index)
    scored_test = pd.DataFrame(index=x_test.index)
    scored_test["Loss_mae"] = np.mean(np.abs(x_pred_test - x_test), axis=1)
    scored_test["Threshold"] = 0.3
    scored_test["Anomaly"] = scored_test["Loss_mae"] > scored_test["Threshold"]
    print(scored_test.head())

    scored_train["Threshold"] = 0.3
    scored_train["Anomaly"] = scored_train["Loss_mae"] > scored_train["Threshold"]
    scored_ae = pd.concat([scored_train, scored_test])
    scored_ae.plot(logy=True, figsize=(10, 6), ylim=[1e-2, 1e2], color=["blue", "red"])
    plt.title("Autoencoder reconstruction MAE vs threshold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
