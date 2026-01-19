import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from DatasetLoader import loader
from DatasetLoader.loader import split_70_15_15, dataframe_to_tensors, build_global_eval_sets

RANDOM_STATE = 42
TARGET_COL = "PINCP"
SENSITIVE_FEATURE = "SEX"  # possible features: SEX, RAC1P

STATES_LIST = [
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
]

CATEGORICAL_COLS = ["AGEP", "COW", "SCHL", "MAR", "RELP", "SEX", "STATE_ID"]
NUMERIC_COLS = ["OCCP", "POBP", "WKHP"]


def load_acs() -> pd.DataFrame:
    """
    Loads ACS state CSVs and applies preprocessing.
    """
    dfs = []
    for state in STATES_LIST:
        try:
            df_state = pd.read_csv(f"./Datasets/acs_dataset/{state}_data.csv")
            df_state["STATE_ID"] = state
            dfs.append(df_state)
        except FileNotFoundError:
            print(f"Warning: File for {state} not found. Skipping.")
            continue

    if not dfs:
        raise ValueError("No ACS state data files found.")

    df = pd.concat(dfs, ignore_index=True)

    if "RAC1P" in df.columns:
        df["RAC1P"] = df["RAC1P"].apply(lambda x: 1 if x == 1 else 0)

    # Global encoding & scaling
    df = loader.encode_categoricals(df, CATEGORICAL_COLS)
    df = loader.scale_numeric_cols(df, NUMERIC_COLS)

    # Shuffle + dropna
    df = shuffle(df, random_state=RANDOM_STATE).dropna()
    return df


def load_acs_states_3():
    """
    NON-IID split: assigns each state to exactly one of 3 clients (random grouping with seed=42).
    Then each client's data is split into 70/15/15 (train/val/test).
    """
    data = load_acs()

    y_encoder = LabelEncoder()
    y_encoder.fit(data[TARGET_COL])

    unique_states = data["STATE_ID"].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(unique_states)
    state_groups = np.array_split(unique_states, 3)

    dfs = [
        data[data["STATE_ID"].isin(state_groups[i])].copy().drop(columns=["STATE_ID"])
        for i in range(3)
    ]

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


def load_acs_states_5():
    """
    NON-IID split: assigns each state to exactly one of 5 clients (random grouping with seed=42).
    Then each client's data is split into 70/15/15 (train/val/test).
    """
    data = load_acs()

    y_encoder = LabelEncoder()
    y_encoder.fit(data[TARGET_COL])

    unique_states = data["STATE_ID"].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(unique_states)
    state_groups = np.array_split(unique_states, 5)

    dfs = [
        data[data["STATE_ID"].isin(state_groups[i])].copy().drop(columns=["STATE_ID"])
        for i in range(5)
    ]

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


def load_acs_random(num_clients: int = 10):
    """
    IID split: breaks up the state structure by shuffling all rows and splitting into N clients.
    Each client is split into 70/15/15 (train/val/test).
    """
    data = load_acs()

    y_encoder = LabelEncoder()
    y_encoder.fit(data[TARGET_COL])

    data = data.drop(columns=["STATE_ID"])

    data = shuffle(data, random_state=RANDOM_STATE)

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

    print(SENSITIVE_FEATURE)
    return (
        data_dict,
        *build_global_eval_sets(
            val_df, test_df,
            target_col=TARGET_COL,
            sensitive_feature=SENSITIVE_FEATURE,
            y_encoder=y_encoder,
        )
    )
