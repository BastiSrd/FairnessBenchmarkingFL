import numpy as np
import pandas as pd
import torch

from loader import (
    DEFAULT_RANDOM_STATE,
    ACS_STATES_LIST,
    split_train_val_test,
    shuffle_dropna,
    fit_label_encoders,
    transform_with_label_encoders,
    fit_standard_scaler,
    transform_with_scaler,
    fit_y_encoder,
    make_federated_dict,
    make_global_split_tensors,
    make_state_groups,
)

ACS_CATEGORICAL_COLUMNS = ["AGEP", "COW", "SCHL", "MAR", "RELP", "SEX", "STATE_ID"]
ACS_NUMERICAL_COLUMNS = ["OCCP", "POBP", "WKHP"]

ACS_TARGET_COL = "PINCP"
ACS_DEFAULT_SENSITIVE = "RAC1P"


def load_acs(dataset_dir: str = "../Datasets/acs_dataset"):
    """
    Loads all state CSVs, merges them into one DataFrame, and applies global preprocessing

    Args:
        dataset_dir (str): Folder containing {state}_data.csv files.

    Returns:
        data (pd.DataFrame): preprocessed merged dataset (includes STATE_ID + target PINCP)
    """

    dfs = []
    for state in ACS_STATES_LIST:
        path = f"{dataset_dir}/{state}_data.csv"
        try:
            df = pd.read_csv(path)
            df["STATE_ID"] = state
            dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: File for {state} not found at {path}. Skipping.")
            continue

    if not dfs:
        raise ValueError("No ACS state CSV files found. Check dataset_dir path.")

    data = pd.concat(dfs, ignore_index=True)

    if ACS_DEFAULT_SENSITIVE in data.columns:
        data[ACS_DEFAULT_SENSITIVE] = data[ACS_DEFAULT_SENSITIVE].apply(lambda x: 1 if x == 1 else 0)

    # Global categorical encoding
    cat_encoders = fit_label_encoders(data, ACS_CATEGORICAL_COLUMNS)
    data = transform_with_label_encoders(data, cat_encoders)

    # Global numerical scaling
    scaler = fit_standard_scaler(data, ACS_NUMERICAL_COLUMNS)
    data = transform_with_scaler(data, scaler, ACS_NUMERICAL_COLUMNS)

    return data


def load_acs_random(
    sensitive_feature: str = ACS_DEFAULT_SENSITIVE,
    num_clients: int = 10,
    dataset_dir: str = "../Datasets/acs_dataset",
):
    """
    Loads ACS and partitions it into N randomly split clients for federated learning (IID-like).

    Args:
        sensitive_feature (str): Column used for fairness grouping (default: 'RAC1P')
        num_clients (int): Number of clients
        dataset_dir (str): Folder containing {state}_data.csv files

    Returns
    -------
    data_dict : dict
        Maps 'client_1'..'client_N' to tensors {X, y, s, y_pot}.
    X_test : torch.Tensor
        Combined test features from all clients.
    y_test : torch.Tensor
        Combined labels for the test set.
    sex_list : list
        Sensitive feature values extracted from the combined test set.
    column_names_list : list
        Feature column names.
    ytest_potential : torch.Tensor
        Potential outcome labels (here: equals y).
    X_val, y_val, sval_list, yval_potential : validation equivalents (global, combined).
    """
    data = load_acs(dataset_dir=dataset_dir)

    # For random split we drop STATE_ID from features
    data = data.drop("STATE_ID", axis=1)

    data = shuffle_dropna(data, random_state=DEFAULT_RANDOM_STATE)

    # global y encoder
    y_encoder = fit_y_encoder(data[ACS_TARGET_COL])

    # split into N chunks
    client_dfs = np.array_split(data, num_clients)

    train_dfs, val_dfs, test_dfs = [], [], []
    for df_chunk in client_dfs:
        df_train, df_val, df_test = split_train_val_test(df_chunk, random_state=DEFAULT_RANDOM_STATE)
        train_dfs.append(df_train)
        val_dfs.append(df_val)
        test_dfs.append(df_test)

    client_names = [f"client_{i}" for i in range(1, num_clients + 1)]
    data_dict = make_federated_dict(
        client_names,
        train_dfs,
        target_col=ACS_TARGET_COL,
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    # Global Val/Test
    X_val, y_val, sval_list, _val_feature_names, yval_potential = make_global_split_tensors(
        val_dfs,
        target_col=ACS_TARGET_COL,
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )
    X_test, y_test, sex_list, column_names_list, ytest_potential = make_global_split_tensors(
        test_dfs,
        target_col=ACS_TARGET_COL,
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    return (
        data_dict,
        X_test, y_test, sex_list, column_names_list, ytest_potential,
        X_val, y_val, sval_list, yval_potential,
    )


def load_acs_states_3(
    sensitive_feature: str = ACS_DEFAULT_SENSITIVE,
    dataset_dir: str = "../Datasets/acs_dataset",
):
    """
    Loads ACS and partitions it into 3 clients by grouping states (non-IID by state).

    Returns (Adult-style):
      (data_dict,
       X_test, y_test, sex_list, column_names_list, ytest_potential,
       X_val, y_val, sval_list, yval_potential)
    """
    data = load_acs(dataset_dir=dataset_dir)
    data = shuffle_dropna(data, random_state=DEFAULT_RANDOM_STATE)

    # Determine state groups deterministically
    unique_states = data["STATE_ID"].unique().tolist()
    state_groups = make_state_groups(unique_states, num_clients=3, seed=DEFAULT_RANDOM_STATE, shuffle_states=True)

    # Build 3 client dfs by state membership
    client_raw = [
        data[data["STATE_ID"].isin(state_groups[0])].copy(),
        data[data["STATE_ID"].isin(state_groups[1])].copy(),
        data[data["STATE_ID"].isin(state_groups[2])].copy(),
    ]

    # global y encoder
    y_encoder = fit_y_encoder(data[ACS_TARGET_COL])

    train_dfs, val_dfs, test_dfs = [], [], []
    for dfc in client_raw:
        df_train, df_val, df_test = split_train_val_test(dfc, random_state=DEFAULT_RANDOM_STATE)

        # Drop STATE_ID after splitting
        train_dfs.append(df_train.drop(["STATE_ID"], axis=1))
        val_dfs.append(df_val.drop(["STATE_ID"], axis=1))
        test_dfs.append(df_test.drop(["STATE_ID"], axis=1))

    client_names = ["client_1", "client_2", "client_3"]
    data_dict = make_federated_dict(
        client_names,
        train_dfs,
        target_col=ACS_TARGET_COL,
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    X_val, y_val, sval_list, _val_feature_names, yval_potential = make_global_split_tensors(
        val_dfs,
        target_col=ACS_TARGET_COL,
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )
    X_test, y_test, sex_list, column_names_list, ytest_potential = make_global_split_tensors(
        test_dfs,
        target_col=ACS_TARGET_COL,
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    return (
        data_dict,
        X_test, y_test, sex_list, column_names_list, ytest_potential,
        X_val, y_val, sval_list, yval_potential,
    )


def load_acs_states_5(
    sensitive_feature: str = ACS_DEFAULT_SENSITIVE,
    dataset_dir: str = "../Datasets/acs_dataset",
):
    """
    Loads ACS and partitions it into 5 clients by grouping states (non-IID by state).

    Returns (Adult-style):
      (data_dict,
       X_test, y_test, sex_list, column_names_list, ytest_potential,
       X_val, y_val, sval_list, yval_potential)
    """
    data = load_acs(dataset_dir=dataset_dir)
    data = shuffle_dropna(data, random_state=DEFAULT_RANDOM_STATE)

    unique_states = data["STATE_ID"].unique().tolist()
    state_groups = make_state_groups(unique_states, num_clients=5, seed=DEFAULT_RANDOM_STATE, shuffle_states=True)

    client_raw = [
        data[data["STATE_ID"].isin(state_groups[i])].copy()
        for i in range(5)
    ]

    y_encoder = fit_y_encoder(data[ACS_TARGET_COL])

    train_dfs, val_dfs, test_dfs = [], [], []
    for dfc in client_raw:
        df_train, df_val, df_test = split_train_val_test(dfc, random_state=DEFAULT_RANDOM_STATE)
        train_dfs.append(df_train.drop(["STATE_ID"], axis=1))
        val_dfs.append(df_val.drop(["STATE_ID"], axis=1))
        test_dfs.append(df_test.drop(["STATE_ID"], axis=1))

    client_names = [f"client_{i}" for i in range(1, 6)]
    data_dict = make_federated_dict(
        client_names,
        train_dfs,
        target_col=ACS_TARGET_COL,
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    X_val, y_val, sval_list, _val_feature_names, yval_potential = make_global_split_tensors(
        val_dfs,
        target_col=ACS_TARGET_COL,
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )
    X_test, y_test, sex_list, column_names_list, ytest_potential = make_global_split_tensors(
        test_dfs,
        target_col=ACS_TARGET_COL,
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    return (
        data_dict,
        X_test, y_test, sex_list, column_names_list, ytest_potential,
        X_val, y_val, sval_list, yval_potential,
    )


if __name__ == "__main__":
    dd, X_test, y_test, s_test, cols, ypot_test, X_val, y_val, s_val, ypot_val = load_acs_states_3()
    print(dd["client_1"]["X"].shape, X_test.shape, X_val.shape)
