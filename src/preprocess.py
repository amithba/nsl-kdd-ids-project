"""
preprocess.py
-------------
Loads NSL-KDD-formatted train/test files, maps fine-grained attack names to
the four standard coarse categories (DoS, Probe, R2L, U2R) + Normal, one-hot
encodes the three categorical features (protocol_type, service, flag),
standard-scales the numeric features, and returns ready-to-train arrays.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from schema import COLUMN_NAMES, CATEGORICAL_COLUMNS, ATTACK_CATEGORY_MAP, CLASSES


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    # Real NSL-KDD files have 43 columns (41 features + label + difficulty).
    # Some mirrors omit the trailing difficulty column -> handle both.
    if df.shape[1] == len(COLUMN_NAMES) - 1:
        df.columns = COLUMN_NAMES[:-1]
        df["difficulty"] = 0
    else:
        df.columns = COLUMN_NAMES[: df.shape[1]]
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.rstrip(".")
    df["category"] = df["label"].map(ATTACK_CATEGORY_MAP)
    unmapped = df["category"].isna()
    if unmapped.any():
        # Any attack name not in our lookup table (e.g. a brand-new unseen
        # attack) is conservatively treated as a generic non-Normal class.
        df.loc[unmapped, "category"] = "DoS"
    return df


def prepare_datasets(train_path: Path, test_path: Path):
    """Returns X_train, X_test, y_train, y_test, feature_names, label_encoder."""
    train_df = map_labels(load_raw(train_path))
    test_df = map_labels(load_raw(test_path))

    train_df = train_df.drop(columns=["label", "difficulty"])
    test_df = test_df.drop(columns=["label", "difficulty"])

    y_train_raw = train_df.pop("category")
    y_test_raw = test_df.pop("category")

    # One-hot encode categoricals; align columns between train/test since the
    # test set can contain service/flag values not seen in training.
    train_oh = pd.get_dummies(train_df, columns=CATEGORICAL_COLUMNS)
    test_oh = pd.get_dummies(test_df, columns=CATEGORICAL_COLUMNS)
    train_oh, test_oh = train_oh.align(test_oh, join="left", axis=1, fill_value=0)
    test_oh = test_oh[train_oh.columns]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_oh.values.astype(float))
    X_test = scaler.transform(test_oh.values.astype(float))

    le = LabelEncoder()
    le.fit(CLASSES)
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    return X_train, X_test, y_train, y_test, list(train_oh.columns), le
