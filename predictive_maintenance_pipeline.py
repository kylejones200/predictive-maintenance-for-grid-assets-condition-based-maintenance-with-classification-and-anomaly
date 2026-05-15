"""Generated from Jupyter notebook: predictive_maintenance_pipeline

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# predictive_maintenance_pipeline.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import WeibullFitter
from scipy.stats import weibull_min
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------
# 1. Load Data
# ---------------------------------------------------


def load_cmapss_data(file_path):
    """Load CMAPSS data from file."""
    df = pd.read_csv(file_path)
    return df


# ---------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------


def preprocess(df, rul_cap=130, drop_cols=None):
    """Compute RUL and health states."""
    df["cycle"] = df["cycle"]

    max_cycle = df.groupby("unit_number")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    df["RUL"] = np.clip(df["RUL"], 0, rul_cap)

    bins = [0, 20, 50, rul_cap]
    labels = ["distress", "warning", "healthy"]
    df["health_state"] = pd.cut(df["RUL"], bins=bins, labels=labels, right=False)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


# ---------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------


def extract_features(df):
    """Extract statistical features per engine."""
    features = df.groupby("unit_number").agg(["mean", "std", "skew", "kurt"])
    features.columns = ["_".join(col) for col in features.columns]
    features.reset_index(inplace=True)
    return features


# ---------------------------------------------------
# 4. Health Index Trend (PCA Reconstruction Error)
# ---------------------------------------------------


def compute_health_index(df, n_components=5):
    """Compute PCA-based health index."""
    feature_cols = [
        col
        for col in df.columns
        if col not in ["unit_number", "cycle", "RUL", "health_state"]
    ]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)

    reconstruction_error = np.mean((X - X_reconstructed) ** 2, axis=1)
    health_index = 1 - (reconstruction_error - reconstruction_error.min()) / (
        reconstruction_error.max() - reconstruction_error.min()
    )

    df["health_index"] = health_index
    return df


# ---------------------------------------------------
# 5. Prepare Labels
# ---------------------------------------------------


def prepare_survival_labels(df):
    """Prepare time-to-event and event labels."""
    df["event"] = 1  # assume failure
    return df[["unit_number", "cycle", "event"]]


def prepare_corn_labels(df):
    """Prepare ordinal labels."""
    label_mapping = {"healthy": 2, "warning": 1, "distress": 0}
    df["ordinal_label"] = df["health_state"].map(label_mapping)
    return df


# ---------------------------------------------------
# 6. Classical Weibull Fitting
# ---------------------------------------------------


def fit_weibull(failure_times):
    """Fit a Weibull distribution to failure times."""
    c, loc, scale = weibull_min.fit(failure_times, floc=0)
    return c, scale


def plot_weibull_survival(c, scale):
    """Plot Weibull survival function."""
    t = np.linspace(0, 1.5 * scale, 100)
    S = np.exp(-((t / scale) ** c))
    plt.plot(t, S)
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Weibull Survival Curve")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------
# 7. Survival Weibull Fitting (with lifelines)
# ---------------------------------------------------


def fit_weibull_survival(df, time_col="cycle", event_col="event"):
    """Fit a parametric Weibull survival model."""
    wf = WeibullFitter()
    wf.fit(df[time_col], event_observed=df[event_col])
    wf.print_summary()
    return wf


# ---------------------------------------------------
# 8. Simple Machine Learning Models
# ---------------------------------------------------


def train_classification(X_train, y_train):
    """Train a classifier."""
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model


def train_regression(X_train, y_train):
    """Train a regressor."""
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model


def evaluate_classification(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Classification Accuracy: {acc:.4f}")
    return acc


def evaluate_regression(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Regression MSE: {mse:.4f}")
    return mse


# ---------------------------------------------------
# 9. Full Pipeline
# ---------------------------------------------------


def run_full_pipeline(file_path):
    df = load_cmapss_data(file_path)
    df = preprocess(df, drop_cols=["setting1", "setting2", "setting3"])  # Optional
    df = compute_health_index(df)

    features = extract_features(df)
    X = features.drop(columns=["unit_number"])

    # Prepare targets
    y_regression = df.groupby("unit_number")["RUL"].first().values
    y_classification = df.groupby("unit_number")["health_state"].first().values
    df = prepare_corn_labels(df)
    y_ordinal = df.groupby("unit_number")["ordinal_label"].first().values
    survival_df = prepare_survival_labels(df)

    # Train/test split
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    _, _, y_train_cls, y_test_cls = train_test_split(
        X, y_classification, test_size=0.2, random_state=42
    )

    # Train models
    print("Training regression model (RUL)...")
    reg_model = train_regression(X_train, y_train_reg)
    evaluate_regression(reg_model, X_test, y_test_reg)

    print("Training classification model (Health State)...")
    cls_model = train_classification(X_train, y_train_cls)
    evaluate_classification(cls_model, X_test, y_test_cls)

    # Fit classical Weibull
    print("Fitting classical Weibull model...")
    failure_times = df.groupby("unit_number")["cycle"].max().values
    c, scale = fit_weibull(failure_times)
    plot_weibull_survival(c, scale)

    # Fit survival Weibull
    print("Fitting survival Weibull model with lifelines...")
    wf_model = fit_weibull_survival(survival_df)

    print("Pipeline complete!")


# ---------------------------------------------------
# Script Execution
# ---------------------------------------------------

if __name__ == "__main__":
    run_full_pipeline("train_FD001.txt")


# --- code cell ---

# !pip install lifelines  # Jupyter-only
