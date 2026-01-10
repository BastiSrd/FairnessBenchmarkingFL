# DatasetLoader/load_acs_data.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from DatasetLoader import loader
from DatasetLoader.loader import split_70_15_15, dataframe_to_tensors, build_global_eval_sets

RANDOM_STATE = 42
TARGET_COL = "PINCP"

STATES_LIST = [
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
]

# In deinem aktuellen Code werden diese Spalten encoded / scaled:
# Note: 'AGEP' wird bei dir encoded (auch wenn es "eigentlich" numerisch ist).
CATEGORICAL_COLS = ["AGEP", "COW", "SCHL", "MAR", "RELP", "SEX", "STATE_ID"]
NUMERIC_COLS = ["OCCP", "POBP", "WKHP"]


def load_acs() -> pd.DataFrame:
    """
    Loads ACS state CSVs and applies full preprocessing:
    - Reads all 50 state files into one DataFrame
    - Adds STATE_ID (state source)
    - Binarizes RAC1P: 1 if White (==1), else 0
    - Encodes categorical columns globally
    - Scales numeric columns globally
    - Shuffles and drops missing values

    Returns
    -------
    df : pandas.DataFrame
        Fully preprocessed ACS DataFrame including STATE_ID for later splitting.
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

    # Sensitive attribute preprocessing (wie bisher):
    # RAC1P -> 1 if White (1), else 0
    if "RAC1P" in df.columns:
        df["RAC1P"] = df["RAC1P"].apply(lambda x: 1 if x == 1 else 0)

    # Global encoding & scaling (wie bisher, nur über loader.py helper aufgerufen)
    df = loader.encode_categoricals(df, CATEGORICAL_COLS)
    df = loader.scale_numeric_cols(df, NUMERIC_COLS)

    # Shuffle + dropna
    df = shuffle(df, random_state=RANDOM_STATE).dropna()
    return df


def load_acs_states_3(sensitive_feature: str = "RAC1P"):
    """
    NON-IID split: assigns each state to exactly one of 3 clients (random grouping with seed=42).
    Then each client's data is split into 70/15/15 (train/val/test).
    """
    data = load_acs()

    # Stabiler y-encoder global (wie bei Adult)
    y_encoder = LabelEncoder()
    y_encoder.fit(data[TARGET_COL])

    unique_states = data["STATE_ID"].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(unique_states)
    state_groups = np.array_split(unique_states, 3)

    # States -> Clients (State bleibt vollständig bei einem Client)
    df1 = data[data["STATE_ID"].isin(state_groups[0])].copy()
    df2 = data[data["STATE_ID"].isin(state_groups[1])].copy()
    df3 = data[data["STATE_ID"].isin(state_groups[2])].copy()

    # STATE_ID ist nur fürs Splitten, nicht als Feature
    df1 = df1.drop(columns=["STATE_ID"])
    df2 = df2.drop(columns=["STATE_ID"])
    df3 = df3.drop(columns=["STATE_ID"])

    df1_tr, df1_val, df1_te = split_70_15_15(df1, seed=RANDOM_STATE)
    df2_tr, df2_val, df2_te = split_70_15_15(df2, seed=RANDOM_STATE)
    df3_tr, df3_val, df3_te = split_70_15_15(df3, seed=RANDOM_STATE)

    X1, y1, s1, ypot1 = dataframe_to_tensors(
        df1_tr, target_col=TARGET_COL, sensitive_feature=sensitive_feature, y_encoder=y_encoder
    )
    X2, y2, s2, ypot2 = dataframe_to_tensors(
        df2_tr, target_col=TARGET_COL, sensitive_feature=sensitive_feature, y_encoder=y_encoder
    )
    X3, y3, s3, ypot3 = dataframe_to_tensors(
        df3_tr, target_col=TARGET_COL, sensitive_feature=sensitive_feature, y_encoder=y_encoder
    )

    data_dict = {
        "client_1": {"X": X1, "y": y1, "s": s1, "y_pot": ypot1},
        "client_2": {"X": X2, "y": y2, "s": s2, "y_pot": ypot2},
        "client_3": {"X": X3, "y": y3, "s": s3, "y_pot": ypot3},
    }

    val_df = pd.concat([df1_val, df2_val, df3_val], ignore_index=True)
    test_df = pd.concat([df1_te, df2_te, df3_te], ignore_index=True)

    return (
        data_dict,
        *build_global_eval_sets(
            val_df, test_df,
            target_col=TARGET_COL,
            sensitive_feature=sensitive_feature,
            y_encoder=y_encoder,
        )
    )


def load_acs_states_5(sensitive_feature: str = "RAC1P"):
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
            tr, target_col=TARGET_COL, sensitive_feature=sensitive_feature, y_encoder=y_encoder
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
            sensitive_feature=sensitive_feature,
            y_encoder=y_encoder,
        )
    )


def load_acs_random(sensitive_feature: str = "RAC1P", num_clients: int = 10):
    """
    IID split: breaks up the state structure by shuffling all rows and splitting into N clients.
    Each client is split into 70/15/15 (train/val/test).
    """
    data = load_acs()

    y_encoder = LabelEncoder()
    y_encoder.fit(data[TARGET_COL])

    # IID: state structure is broken → STATE_ID removed from features
    data = data.drop(columns=["STATE_ID"])

    # (Optional) extra shuffle here keeps behavior close to your previous code
    data = shuffle(data, random_state=RANDOM_STATE)

    client_dfs = np.array_split(data, num_clients)

    data_dict = {}
    val_parts, test_parts = [], []

    for i, df_chunk in enumerate(client_dfs, start=1):
        tr, va, te = split_70_15_15(df_chunk, seed=RANDOM_STATE)
        X, y, s, ypot = dataframe_to_tensors(
            tr, target_col=TARGET_COL, sensitive_feature=sensitive_feature, y_encoder=y_encoder
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
            sensitive_feature=sensitive_feature,
            y_encoder=y_encoder,
        )
    )
