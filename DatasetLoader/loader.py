# loader.py
"""
Shared utilities + constants for all dataset loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle

# Shared constants

ACS_STATES_LIST: List[str] = [
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
]

CAC_STATE_MAP_3: Dict[str, List[int]] = {
    "client_1": [38, 44, 56, 32, 35, 28, 54, 41, 6, 33, 51, 10, 37, 12, 27, 8],
    "client_2": [24, 47, 22, 48, 36, 40, 34, 50, 13, 5, 49, 53, 23, 39, 9],
    "client_3": [21, 16, 29, 25, 45, 1, 18, 20, 55, 4, 46, 42, 2, 19, 11],
}

CAC_STATE_MAP_5: Dict[str, List[int]] = {
    "client_1": [38, 44, 56, 32, 35, 28, 54, 41, 6, 33],
    "client_2": [51, 10, 37, 12, 27, 8, 24, 47, 22],
    "client_3": [48, 36, 40, 34, 50, 13, 5, 49, 53],
    "client_4": [23, 39, 9, 21, 16, 29, 25, 45, 1],
    "client_5": [18, 20, 55, 4, 46, 42, 2, 19, 11],
}

DEFAULT_RANDOM_STATE: int = 42

@dataclass
class EncodersBundle:
    """
    Holds fitted encoders/scalers for reuse (fit once, transform many times).
    """
    cat_encoders: Dict[str, LabelEncoder]
    scaler: Optional[StandardScaler] = None
    y_encoder: Optional[LabelEncoder] = None

def split_train_val_test(
    df: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into train/val/test.

    Input:
      df: DataFrame

    Return:
      train_df, val_df, test_df
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    # first split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df, test_size=(1.0 - train_ratio), random_state=random_state
    )

    # second split: val vs test (split temp evenly if val==test)
    if val_ratio + test_ratio == 0:
        raise ValueError("val_ratio and test_ratio cannot both be 0")

    # proportion of temp going to test:
    test_share_of_temp = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=test_share_of_temp, random_state=random_state
    )

    return train_df, val_df, test_df


def shuffle_dropna(
    df: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE,
    subset: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Shuffles a dataframe deterministically and drops NaNs.

    Input:
      df: DataFrame
      subset: optional columns to consider for dropna()

    Return:
      shuffled & cleaned DataFrame
    """
    df2 = shuffle(df, random_state=random_state)
    if subset is None:
        return df2.dropna()
    return df2.dropna(subset=list(subset))

# Encoding / scaling helpers

def fit_label_encoders(
    df: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> Dict[str, LabelEncoder]:
    """
    Fits LabelEncoders for each categorical column on the given df.

    Input:
      df: DataFrame
      categorical_columns: columns to encode

    Return:
      dict col -> fitted LabelEncoder
    """

    encoders: Dict[str, LabelEncoder] = {}
    for col in categorical_columns:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le
    return encoders


def transform_with_label_encoders(
    df: pd.DataFrame,
    encoders: Mapping[str, LabelEncoder],
) -> pd.DataFrame:
    """
    Applies already-fitted label encoders to df.

    Input:
      df: DataFrame
      encoders: dict col -> LabelEncoder

    Return:
      encoded copy of df
    """
    out = df.copy()
    for col, le in encoders.items():
        if col in out.columns:
            out[col] = le.transform(out[col])
    return out


def fit_standard_scaler(
    df: pd.DataFrame,
    numerical_columns: Sequence[str],
) -> StandardScaler:
    """
    Fits a StandardScaler on numerical columns (on provided df).

    Return:
      fitted scaler
    """
    cols = [c for c in numerical_columns if c in df.columns]
    scaler = StandardScaler()
    if cols:
        scaler.fit(df[cols])
    return scaler


def transform_with_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    numerical_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Applies an already-fitted StandardScaler to df.

    Return:
      scaled copy of df
    """
    out = df.copy()
    cols = [c for c in numerical_columns if c in out.columns]
    if cols:
        out[cols] = scaler.transform(out[cols])
    return out


def fit_y_encoder(
    y_series: pd.Series,
) -> LabelEncoder:
    """
    Fits a LabelEncoder for the target column (classification).
    """
    le = LabelEncoder()
    le.fit(y_series)
    return le

# Tensor conversion helpers

def make_client_tensors(
    df_split: pd.DataFrame,
    target_col: str,
    sensitive_feature: str,
    *,
    y_encoder: Optional[LabelEncoder] = None,
    x_dtype: torch.dtype = torch.float32,
    y_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a split dataframe into tensors for FL.

    Input:
      df_split: DataFrame containing target_col + feature columns
      target_col: name of label/target column
      sensitive_feature: column name inside X used as sensitive attribute
      y_encoder: optional LabelEncoder for classification target (for consistent mapping)
      task: "classification" or "regression"

    Return:
      X_t: torch.FloatTensor (N, D)
      y_t: torch.FloatTensor (N,)
      s_t: torch.FloatTensor (N,)
      y_pot_t: torch.FloatTensor (N,)  (currently same as y)
    """
    if target_col not in df_split.columns:
        raise KeyError(f"target_col='{target_col}' not found in df_split columns")

    X_df = df_split.drop(columns=[target_col])

    if sensitive_feature not in X_df.columns:
        raise KeyError(f"sensitive_feature='{sensitive_feature}' not found in feature columns")

    y_np = y_encoder.transform(df_split[target_col]).astype(np.float32, copy=False)

    s_np = X_df[sensitive_feature].to_numpy(dtype=np.float32, copy=False)

    y_pot_np = y_np

    X_t = torch.tensor(X_df.values, dtype=x_dtype)
    y_t = torch.tensor(y_np, dtype=y_dtype)
    s_t = torch.tensor(s_np, dtype=torch.float32)
    y_pot_t = torch.tensor(y_pot_np, dtype=y_dtype)

    return X_t, y_t, s_t, y_pot_t


def make_federated_dict(
    client_names: Sequence[str],
    client_splits: Sequence[pd.DataFrame],
    *,
    target_col: str,
    sensitive_feature: str,
    y_encoder: Optional[LabelEncoder] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Builds the standard data_dict used by your code:

      data_dict["client_i"] = {"X": X, "y": y, "s": s, "y_pot": y_pot}

    Input:
      client_names: e.g. ["client_1", "client_2", ...]
      client_splits: list of DataFrames (same length)
      target_col, sensitive_feature, y_encoder, task

    Return:
      data_dict
    """

    if len(client_names) != len(client_splits):
        raise ValueError("client_names and client_splits must have the same length")

    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, df_split in zip(client_names, client_splits):
        X, y, s, y_pot = make_client_tensors(
            df_split,
            target_col=target_col,
            sensitive_feature=sensitive_feature,
            y_encoder=y_encoder,
        )
        out[name] = {"X": X, "y": y, "s": s, "y_pot": y_pot}
    return out


def make_global_split_tensors(
    dfs: Sequence[pd.DataFrame],
    *,
    target_col: str,
    sensitive_feature: str,
    y_encoder: Optional[LabelEncoder] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[Any], List[str], torch.Tensor]:
    """
    Concatenates multiple split-DFs (e.g. val or test from each client),
    then returns global tensors + sensitive raw list + feature names + y_pot.

    Return:
      X_global, y_global, s_raw_list, feature_names, y_pot_global
    """

    if not dfs:
        raise ValueError("dfs must not be empty")

    merged = pd.concat(list(dfs), ignore_index=True)

    X_df = merged.drop(columns=[target_col])
    feature_names = X_df.columns.tolist()
    s_raw_list = X_df[sensitive_feature].tolist()

    y_np = y_encoder.transform(merged[target_col]).astype(np.float32, copy=False)

    X_t = torch.tensor(X_df.values, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)
    y_pot_t = torch.tensor(y_np, dtype=torch.float32)

    return X_t, y_t, s_raw_list, feature_names, y_pot_t

# Deterministic state/client mapping helpers

def make_state_groups(
    states: Sequence[Any],
    num_clients: int,
    *,
    seed: int = DEFAULT_RANDOM_STATE,
    shuffle_states: bool = True,
) -> List[List[Any]]:
    """
    Deterministically split a list of states into N groups (for ACS-style non-IID by states).

    Input:
      states: e.g. list of state abbreviations or state IDs
      num_clients: number of groups
      seed: random seed for stable grouping
      shuffle_states: if True, shuffle order before splitting

    Return:
      list of groups, length=num_clients, each a list of states
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")

    arr = np.array(list(states), dtype=object)
    if shuffle_states:
        rng = np.random.RandomState(seed)
        rng.shuffle(arr)

    groups = np.array_split(arr, num_clients)
    return [g.tolist() for g in groups]


def make_state_map(
    states: Sequence[Any],
    num_clients: int,
    *,
    seed: int = DEFAULT_RANDOM_STATE,
    shuffle_states: bool = True,
    client_prefix: str = "client_",
) -> Dict[str, List[Any]]:
    """
    Returns a dict mapping client names to a deterministic state group list.

    Example:
      make_state_map(ACS_STATES_LIST, 3) -> {"client_1":[...], "client_2":[...], "client_3":[...]}
    """
    groups = make_state_groups(
        states=states,
        num_clients=num_clients,
        seed=seed,
        shuffle_states=shuffle_states,
    )
    return {f"{client_prefix}{i+1}": grp for i, grp in enumerate(groups)}
