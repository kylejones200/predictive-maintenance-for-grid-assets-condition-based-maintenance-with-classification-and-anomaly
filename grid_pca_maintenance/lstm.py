"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Input, Layer
from tensorflow.keras.models import Model, Sequential
from tqdm import tqdm

def lstm() -> None:
    plt.figure(figsize=(10, 4))

    plt.plot(cycles, preds_plain, label="LSTM")

    plt.axhline(0.5, color="red", linestyle="--", label="Threshold")

    plt.title(f"LSTM Distress Prediction for Engine {unit_id}")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("lstm_engine_prediction.png")

    plt.show()

    plt.figure(figsize=(10, 4))

    plt.plot(cycles, preds_attn, label="LSTM + Attention")

    plt.axhline(0.5, color="red", linestyle="--", label="Threshold")

    plt.title(f"LSTM + Attention Prediction for Engine {unit_id}")

    plt.xlabel("Cycle")

    plt.ylabel("Distress Probability")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("lstm_attention_engine_prediction.png")

    plt.show()

