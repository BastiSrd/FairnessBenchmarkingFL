from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle

from DatasetLoader import loader
from DatasetLoader.loader import split_70_15_15, dataframe_to_tensors, build_global_eval_sets

RANDOM_STATE = 42
TARGET_COL = "income"
SENSITIVE_FEATURE = "sex"  # possible features: sex

BASE_DIR = Path(__file__).resolve().parent.parent
ADULT_PATH = BASE_DIR / "Datasets" / "adult.csv"

CATEGORICAL_COLS = [
    "workclass", "education", "marital.status", "occupation",
    "relationship", "race", "sex", "native.country",
]

NUMERIC_COLS_ALL = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
NUMERIC_COLS_EXCL_AGE = ["fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]


def _scale_age_per_client(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales age for each client.
    """
    age_scaler = StandardScaler()
    df = df.copy()
    df["age"] = age_scaler.fit_transform(df[["age"]]).ravel().astype("float32")
    return df


def load_adult(*, for_iid: bool) -> pd.DataFrame:
    """
    Loads ACS state CSVs and applies preprocessing.
    """
    df = pd.read_csv(ADULT_PATH)

    df = loader.encode_categoricals(df, CATEGORICAL_COLS)

    if for_iid:
        df = loader.scale_numeric_cols(df, NUMERIC_COLS_ALL)
    else:
        df = loader.scale_numeric_cols(df, NUMERIC_COLS_EXCL_AGE)

    df = shuffle(df, random_state=RANDOM_STATE).dropna()
    return df


def load_adult_age3():
    """
    NON-IID split: assigns age groups to exactly one of 3 clients (random grouping with seed=42).
    Then each client's data is split into 70/15/15 (train/val/test).
    """
    data = load_adult(for_iid=False)

    y_encoder = LabelEncoder()
    y_encoder.fit(data[TARGET_COL])

    df1 = data[(data["age"] >= 0) & (data["age"] <= 29)].copy()
    df2 = data[(data["age"] >= 30) & (data["age"] <= 39)].copy()
    df3 = data[(data["age"] >= 40)].copy()

    dfs = [_scale_age_per_client(d) for d in (df1, df2, df3)]
    splits = [split_70_15_15(d, seed=RANDOM_STATE) for d in dfs]

    data_dict = {}
    val_parts, test_parts = [], []
    for i, (tr, va, te) in enumerate(splits, start=1):
        X, y, s, ypot = dataframe_to_tensors(
            tr, target_col=TARGET_COL, sensitive_feature=SENSITIVE_FEATURE, y_encoder=y_encoder
        )
        data_dict[f"client_{i}"] = {"X": X, "y": y, "s": s, "y_pot": ypot}
        val_parts.append(va)
        test_parts.append(te)

    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return (
        data_dict,
        *build_global_eval_sets(
            val_df, test_df,
            target_col=TARGET_COL,
            sensitive_feature=SENSITIVE_FEATURE,
            y_encoder=y_encoder,
        )
    )


def load_adult_age5():
    """
    NON-IID split: assigns age groups to exactly one of 5 clients (random grouping with seed=42).
    Then each client's data is split into 70/15/15 (train/val/test).
    """
    data = load_adult(for_iid=False)

    y_encoder = LabelEncoder()
    y_encoder.fit(data[TARGET_COL])

    df1 = data[(data["age"] >= 0) & (data["age"] <= 30)].copy()
    df2 = data[(data["age"] >= 31) & (data["age"] <= 35)].copy()
    df3 = data[(data["age"] >= 36) & (data["age"] <= 45)].copy()
    df4 = data[(data["age"] >= 46) & (data["age"] <= 55)].copy()
    df5 = data[(data["age"] >= 56)].copy()

    dfs = [_scale_age_per_client(d) for d in (df1, df2, df3, df4, df5)]
    splits = [split_70_15_15(d, seed=RANDOM_STATE) for d in dfs]

    data_dict = {}
    val_parts, test_parts = [], []
    for i, (tr, va, te) in enumerate(splits, start=1):
        X, y, s, ypot = dataframe_to_tensors(
            tr, target_col=TARGET_COL, sensitive_feature=SENSITIVE_FEATURE, y_encoder=y_encoder
        )
        data_dict[f"client_{i}"] = {"X": X, "y": y, "s": s, "y_pot": ypot}
        val_parts.append(va)
        test_parts.append(te)

    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return (
        data_dict,
        *build_global_eval_sets(
            val_df, test_df,
            target_col=TARGET_COL,
            sensitive_feature=SENSITIVE_FEATURE,
            y_encoder=y_encoder,
        )
    )


def load_adult_random(num_clients: int = 10):
    """
    IID split: breaks up the age groups by splitting into N clients.
    Each client is split into 70/15/15 (train/val/test).
    """
    data = load_adult(for_iid=True)

    y_encoder = LabelEncoder()
    y_encoder.fit(data[TARGET_COL])

    client_dfs = np.array_split(data, num_clients)

    data_dict = {}
    val_parts, test_parts = [], []

    for i, df_chunk in enumerate(client_dfs, start=1):
        tr, va, te = split_70_15_15(df_chunk, seed=RANDOM_STATE)
        X, y, s, ypot = dataframe_to_tensors(
            tr, target_col=TARGET_COL, sensitive_feature=SENSITIVE_FEATURE, y_encoder=y_encoder
        )
        data_dict[f"client_{i}"] = {"X": X, "y": y, "s": s, "y_pot": ypot}
        val_parts.append(va)
        test_parts.append(te)

    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return (
        data_dict,
        *build_global_eval_sets(
            val_df, test_df,
            target_col=TARGET_COL,
            sensitive_feature=SENSITIVE_FEATURE,
            y_encoder=y_encoder,
        )
    )
