from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle

from DatasetLoader import loader
from DatasetLoader.loader import split_70_15_15, dataframe_to_tensors, build_global_eval_sets

BASE_DIR = Path(__file__).resolve().parent.parent
BANK_PATH = BASE_DIR / "Datasets" / "bank-full.csv"

RANDOM_STATE = 42
TARGET_COL = "y"

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "previous", "poutcome"
]

NUMERIC_COLS_EXCL_AGE = ["balance", "day", "duration", "campaign", "pdays"]
NUMERIC_COLS_ALL = ["age", "balance", "day", "duration", "campaign", "pdays"]


def _scale_age_per_client(df: pd.DataFrame) -> pd.DataFrame:
    """
    WICHTIG: wie in deinem bisherigen Code wird 'age' pro Client-Gruppe separat skaliert
    """
    age_scaler = StandardScaler()
    df = df.copy()
    df["age"] = age_scaler.fit_transform(df[["age"]]).ravel().astype("float32")
    return df


def load_bank(*, for_iid: bool) -> pd.DataFrame:
    """
    Loads the Bank dataset and applies the full preprocessing.

    Parameters
    ----------
    url : str
        Path to bank-full.csv
    for_iid : bool
        If True: preprocess for IID split (scale all numerics incl. age)
        If False: preprocess for NON-IID split (scale numerics excl. age)

    Returns
    -------
    df : pandas.DataFrame
        Fully preprocessed DataFrame.
    """
    df = pd.read_csv(BANK_PATH)

    # Encode categoricals (dataset-specific list)
    df = loader.encode_categoricals(df, CATEGORICAL_COLS)

    # Scale numerics depending on scenario
    if for_iid:
        df = loader.scale_numeric_cols(df, NUMERIC_COLS_ALL)
    else:
        df = loader.scale_numeric_cols(df, NUMERIC_COLS_EXCL_AGE)

    # Shuffle + dropna
    df = shuffle(df, random_state=RANDOM_STATE).dropna()
    return df


def load_bank_age_5(sensitive_feature):
    """
    Loads Bank dataset and partitions it into five age-based clients for federated learning.
    The y_pot's are currently set to the exact same data as y.

    Args:
        url (str): The web link or local path to the Bank dataset CSV file.
        sensitive_feature (str): The column name to be used as the sensitive attribute.

    Returns
    -------
    data_dict : dict
        Maps 'client_1' through 'client_5' (stratified by age) to tensors {X, y, s, y_pot}.
    X_test : torch.Tensor
        Combined and normalized test features from all five age partitions.
    y_test : torch.Tensor
        Combined binary labels for the test set.
    sex_list : list
        Sensitive feature values extracted from the combined test set.
    column_names_list : list
        List of strings representing the feature column names.
    ytest_potential : torch.Tensor
        Potential outcome labels for the combined test set.
    """
    data = load_bank(for_iid=False)

    y_encoder = LabelEncoder()
    y_encoder.fit(data[TARGET_COL])

    # exakte Age-Gruppen wie bisher
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


def load_bank_age3(sensitive_feature):
    """
    Loads Bank dataset and partitions it into three age-based clients for federated learning.
    The y_pot's are currently set to the exact same data as y.

    Args:
        sensitive_feature (str): The column name to be used as the sensitive attribute.

    Returns
    -------
    data_dict : dict
        Maps 'client_1' (0-29), 'client_2' (30-39), and 'client_3' (40+) to tensors {X, y, s, y_pot}.
    X_test : torch.Tensor
        Combined and normalized test features from all three age partitions.
    y_test : torch.Tensor
        Combined binary labels for the test set.
    sex_list : list
        Sensitive feature values extracted from the combined test set.
    column_names_list : list
        List of strings representing the feature column names.
    ytest_potential : torch.Tensor
        Potential outcome labels for the combined test set.
    """
    data = load_bank(for_iid=False)

    y_encoder = LabelEncoder()
    y_encoder.fit(data[TARGET_COL])

    # exakte Age-Gruppen wie bisher
    df1 = data[(data["age"] >= 0) & (data["age"] <= 29)].copy()
    df2 = data[(data["age"] >= 30) & (data["age"] <= 39)].copy()
    df3 = data[(data["age"] >= 40)].copy()

    # Age separat pro Client skalieren (wie vorher)
    df1 = _scale_age_per_client(df1)
    df2 = _scale_age_per_client(df2)
    df3 = _scale_age_per_client(df3)

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


def load_bank_random(sensitive_feature, num_clients):
    """
    Loads Bank dataset and partitions it into N randomly split clients for federated learning.

    Args:
        url (str): The web link or local path to the Bank dataset CSV file.
        sensitive_feature (str): The column name to be used as the sensitive attribute.
        num_clients (int): The number of clients to split the data into.

    Returns
    -------
    data_dict : dict
        Maps 'client_1' through 'client_N' to tensors {X, y, s, y_pot}.
    X_test : torch.Tensor
        Combined and normalized test features from all partitions.
    y_test : torch.Tensor
        Combined binary labels for the test set.
    sex_list : list
        Sensitive feature values extracted from the combined test set.
    column_names_list : list
        List of strings representing the feature column names.
    ytest_potential : torch.Tensor
        Potential outcome labels for the combined test set.
    """
    data = load_bank(for_iid=True)

    y_encoder = LabelEncoder()
    y_encoder.fit(data[TARGET_COL])

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


if __name__ == "__main__":
    a, _, _, _, _, _ = load_bank_random("../Datasets/bank-full.csv", "age", 7)
    print(a)
