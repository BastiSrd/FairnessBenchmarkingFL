from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

ALLOWED_SENSITIVE = {
    "racepctblack",
    "racePctWhite",
    "racePctAsian",
    "racePctHisp",
    "PctForeignBorn",
}

TARGET_COL = "ViolentCrimesPerPop"

# Clients are formed by a random but fixed partition of states.
CAC_STATE_MAP_3 = {
    "client_1": [38, 44, 56, 32, 35, 28, 54, 41, 6, 33, 51, 10, 37, 12, 27, 8],
    "client_2": [24, 47, 22, 48, 36, 40, 34, 50, 13, 5, 49, 53, 23, 39, 9],
    "client_3": [21, 16, 29, 25, 45, 1, 18, 20, 55, 4, 46, 42, 2, 19, 11],
}

CAC_STATE_MAP_5 = {
    "client_1": [38, 44, 56, 32, 35, 28, 54, 41, 6, 33],
    "client_2": [51, 10, 37, 12, 27, 8, 24, 47, 22],
    "client_3": [48, 36, 40, 34, 50, 13, 5, 49, 53],
    "client_4": [23, 39, 9, 21, 16, 29, 25, 45, 1],
    "client_5": [18, 20, 55, 4, 46, 42, 2, 19, 11],
}

BASE_DIR = next(
    p for p in Path(__file__).resolve().parents
    if (p / "Datasets").exists()
)


def _parse_uci_names_file(names_path: str):
    """
    Extracts attribute names from the UCI 'communities.names' file.
    Looks for lines like: '@attribute <name> <type>'
    """
    cols = []
    with open(names_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("@attribute"):
                parts = line.split()
                # @attribute NAME TYPE
                if len(parts) >= 2:
                    cols.append(parts[1])
    return cols


def communities_data_to_csv(data_path: str, names_path: str, out_csv_path: str):
    """
    Converts the raw UCI Communities & Crime .data file into a CSV with proper column names.

    Args:
        data_path (str): Path to the raw 'communities.data' file (comma-separated, no header).
        names_path (str): Path to the 'communities.names' file containing @attribute definitions.
        out_csv_path (str): Output path where the generated CSV file will be saved.

    Returns:
        pd.DataFrame:
            The full Communities & Crime dataset as a DataFrame with column headers and NaNs for missing values.
    """
    cols = _parse_uci_names_file(names_path)
    df = pd.read_csv(data_path, header=None, names=cols, na_values=["?"])
    df.to_csv(out_csv_path, index=False)
    return df


def load_cac(url):
    """
    Loads and preprocesses the Communities & Crime dataset for centralized regression.

    Args:
        url (str): Path to the preprocessed Communities & Crime CSV file.

    Returns:
        X (pd.DataFrame):
            Standardized feature matrix containing all numeric attributes except the target.
        y (np.ndarray):
            Regression target array containing ViolentCrimesPerPop values (float32).
    """
    data = pd.read_csv(url)

    # Drop non-numeric text column if present
    if "communityname" in data.columns:
        data = data.drop(columns=["communityname"])

    # Convert '?' etc. to NaN and ensure numeric
    data = data.apply(pd.to_numeric, errors="coerce").dropna()

    # Standardize all feature columns (everything except target)
    feature_cols = [c for c in data.columns if c != TARGET_COL]
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])

    X = data.drop(TARGET_COL, axis=1)
    y = data[TARGET_COL].to_numpy(dtype=np.float32)
    return X, y


def load_cac_random(url, sensitive_feature, num_clients):
    """
    Loads the Communities & Crime dataset and creates an IID federated split into N random clients.

    Args:
        url (str): Path to the Communities & Crime CSV file.
        sensitive_feature (str): Name of the sensitive attribute used for fairness evaluation.
        num_clients (int): Number of federated clients to create.

    Returns:
        data_dict (dict):
            Dictionary mapping 'client_i' to tensors {X, y, s, y_pot}.
        X_test (torch.Tensor):
            Combined test feature tensor from all clients.
        y_test (torch.Tensor):
            Combined regression targets for the test set.
        sensitive_list (list):
            Sensitive feature values corresponding to the global test set.
        column_names_list (list):
            Names of all feature columns used for training.
        ytest_potential (torch.Tensor):
            Placeholder tensor for potential outcomes (same shape as y_test).
    """
    if sensitive_feature not in ALLOWED_SENSITIVE:
        raise ValueError(f"sensitive_feature must be one of {sorted(ALLOWED_SENSITIVE)}")

    data = pd.read_csv(url)

    if "communityname" in data.columns:
        data = data.drop(columns=["communityname"])

    data = data.apply(pd.to_numeric, errors="coerce").dropna()

    # Standardize all feature columns (exclude target)
    feature_cols = [c for c in data.columns if c != TARGET_COL]
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])

    # Shuffle (IID)
    data = shuffle(data, random_state=42)

    # Split into N chunks
    client_dfs = [
        data.iloc[i::num_clients].reset_index(drop=True)
        for i in range(num_clients)
    ]

    data_dict = {}
    test_dfs_list = []

    for i, df_chunk in enumerate(client_dfs):
        client_name = f"client_{i + 1}"

        df_chunk = df_chunk.dropna()
        df_train, df_test = train_test_split(df_chunk, test_size=0.1, random_state=42)
        test_dfs_list.append(df_test)

        X_client = df_train.drop(TARGET_COL, axis=1)
        y_client = df_train[TARGET_COL].to_numpy(dtype=np.float32)

        s_client = X_client[sensitive_feature]
        y_potential_client = y_client
        X_tensor = torch.tensor(X_client.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_client, dtype=torch.float32)
        s_tensor = torch.from_numpy(s_client.values).float()
        y_pot_tensor = torch.tensor(y_potential_client, dtype=torch.float32)

        data_dict[client_name] = {"X": X_tensor, "y": y_tensor, "s": s_tensor, "y_pot": y_pot_tensor}

    # Global test set
    test_df = pd.concat(test_dfs_list, ignore_index=True)

    X_test = test_df.drop(TARGET_COL, axis=1)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.float32)

    sex_column = X_test[sensitive_feature]
    column_names_list = X_test.columns.tolist()
    sex_list = sex_column.tolist()

    ytest_potential = y_test
    ytest_potential_tensor = torch.tensor(ytest_potential, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return data_dict, X_test_tensor, y_test_tensor, sex_list, column_names_list, ytest_potential_tensor


def _load_cac_noniid_state(url, sensitive_feature, state_map):
    """
        Internal helper that creates a non-IID federated split by assigning disjoint sets of states to clients.

        Args:
            url (str): Path to the Communities & Crime CSV file.
            sensitive_feature (str): Name of the sensitive attribute used for fairness evaluation.
            state_map (dict):
                Mapping from client names (e.g., 'client_1') to lists of state codes.

        Returns:
            data_dict (dict):
                Dictionary mapping each client to tensors {X, y, s, y_pot}.
            X_test (torch.Tensor):
                Combined test feature tensor from all clients.
            y_test (torch.Tensor):
                Combined regression targets for the test set.
            sensitive_list (list):
                Sensitive feature values corresponding to the global test set.
            column_names_list (list):
                Names of all feature columns used for training.
            ytest_potential (torch.Tensor):
                Placeholder tensor for potential outcomes (same shape as y_test).
        """
    if sensitive_feature not in ALLOWED_SENSITIVE:
        raise ValueError(f"sensitive_feature must be one of {sorted(ALLOWED_SENSITIVE)}")

    data = pd.read_csv(url)

    if "communityname" in data.columns:
        data = data.drop(columns=["communityname"])

    # numeric + drop missing
    data = data.apply(pd.to_numeric, errors="coerce").dropna()

    data["state"] = data["state"].astype(int)

    # scale all cols except target and state
    feature_cols = [c for c in data.columns if c != TARGET_COL]
    scale_cols = [c for c in feature_cols if c != "state"]
    scaler = StandardScaler()
    data[scale_cols] = scaler.fit_transform(data[scale_cols])

    data = shuffle(data, random_state=42)

    client_names = sorted(state_map.keys(), key=lambda x: int(x.split("_")[1]))
    dfs = []
    test_dfs = []

    for cname in client_names:
        dfc = data[data["state"].isin(state_map[cname])].dropna()

        if len(dfc) == 0:
            raise RuntimeError(f"{cname} got 0 rows. Check your CAC_STATE_MAP_* and 'state' values.")

        dfc, test_dfc = train_test_split(dfc, test_size=0.1, random_state=42)
        dfs.append((cname, dfc))
        test_dfs.append(test_dfc)

    test_df = pd.concat(test_dfs, ignore_index=True)

    data_dict = {}

    for cname, dfc in dfs:
        X_client = dfc.drop([TARGET_COL, "state"], axis=1)
        y_client = dfc[TARGET_COL].to_numpy(dtype=np.float32)

        s_client = X_client[sensitive_feature]
        y_potential_client = y_client

        X_client = torch.tensor(X_client.values, dtype=torch.float32)
        y_client = torch.tensor(y_client, dtype=torch.float32)
        s_client = torch.from_numpy(s_client.values).float()
        y_potential_client = torch.tensor(y_potential_client, dtype=torch.float32)

        data_dict[cname] = {"X": X_client, "y": y_client, "s": s_client, "y_pot": y_potential_client}

    X_test = test_df.drop([TARGET_COL, "state"], axis=1)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.float32)

    sensitive_column = X_test[sensitive_feature]
    column_names_list = X_test.columns.tolist()
    sensitive_list = sensitive_column.tolist()

    ytest_potential = torch.tensor(y_test, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return data_dict, X_test, y_test, sensitive_list, column_names_list, ytest_potential


def load_cac_noniid_3(url, sensitive_feature):
    """
       Loads the Communities & Crime dataset and creates a non-IID federated split into 3 clients based on state assignment.

       Args:
           url (str): Path to the Communities & Crime CSV file.
           sensitive_feature (str): Name of the sensitive attribute used for fairness evaluation.

       Returns:
           data_dict (dict):
               Dictionary mapping 'client_1' to 'client_3' to tensors {X, y, s, y_pot}.
           X_test (torch.Tensor):
               Combined test feature tensor from all clients.
           y_test (torch.Tensor):
               Combined regression targets for the test set.
           sensitive_list (list):
               Sensitive feature values corresponding to the global test set.
           column_names_list (list):
               Names of all feature columns used for training.
           ytest_potential (torch.Tensor):
               Placeholder tensor for potential outcomes (same shape as y_test).
       """

    return _load_cac_noniid_state(url, sensitive_feature, CAC_STATE_MAP_3)


def load_cac_noniid_5(url, sensitive_feature):
    """
       Loads the Communities & Crime dataset and creates a non-IID federated split into 5 clients based on state assignment.

       Args:
           url (str): Path to the Communities & Crime CSV file.
           sensitive_feature (str): Name of the sensitive attribute used for fairness evaluation.

       Returns:
           data_dict (dict):
               Dictionary mapping 'client_1' to 'client_5' to tensors {X, y, s, y_pot}.
           X_test (torch.Tensor):
               Combined test feature tensor from all clients.
           y_test (torch.Tensor):
               Combined regression targets for the test set.
           sensitive_list (list):
               Sensitive feature values corresponding to the global test set.
           column_names_list (list):
               Names of all feature columns used for training.
           ytest_potential (torch.Tensor):
               Placeholder tensor for potential outcomes (same shape as y_test).
       """

    return _load_cac_noniid_state(url, sensitive_feature, CAC_STATE_MAP_5)


if __name__ == "__main__":
    df = communities_data_to_csv(
        data_path=BASE_DIR / "Datasets" / "cac_dataset" / "communities.data",
        names_path=BASE_DIR / "Datasets" / "cac_dataset" / "communities.names",
        out_csv_path=BASE_DIR / "Datasets" / "communities.csv"
    )

    a, _, _, _, _, _ = load_cac_random(
        BASE_DIR / "Datasets" / "communities.csv",
        "PctForeignBorn",
        7
    )
    print(a)

    b, _, _, _, _, _ = load_cac_noniid_3(
        BASE_DIR / "Datasets" / "communities.csv",
        "racepctblack"
    )
    print(b)
