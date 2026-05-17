"""Notebook steps (auto-split)."""

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

def pseudo_code_for_llm_interaction() -> None:
    prompt = "\nYou are looking at a time series forecasting model for sensor_9 from a jet engine.\nThe model starts to deviate significantly from actual values after cycle 150.\nWhat might cause this, and what could you try to improve it?\n"

    response = openai.ChatCompletion.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )

    print(response["choices"][0]["message"]["content"])

