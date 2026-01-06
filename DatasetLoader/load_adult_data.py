import torch
from psmpy.plotting import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle

from loader import (
    DEFAULT_RANDOM_STATE,
    split_train_val_test,
    shuffle_dropna,
    fit_label_encoders,
    transform_with_label_encoders,
    fit_standard_scaler,
    transform_with_scaler,
    fit_y_encoder,
    make_client_tensors,
    make_federated_dict,
    make_global_split_tensors,
)

ADULT_CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital.status", "occupation",
    "relationship", "race", "sex", "native.country",
]

ADULT_NUMERICAL_COLUMNS_ALL = [
    "age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"
]


ADULT_NUMERICAL_COLUMNS_EXCL_AGE = [
    "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"
]

def load_adult(url):
    """
    Loads, encodes, and normalizes the Adult Census Income dataset from a URL.

    Args:
        url (str): The web link or local path to the Adult dataset CSV file.

    Returns
    -------
        X (pd.DataFrame): Processed feature matrix with encoded categories and scaled numerical values.
        y (numpy.ndarray): Encoded target vector (income) where 0 and 1 represent income brackets.
    """
    data = pd.read_csv(url)
    # data = shuffle(data)

    # Encode categorical columns (fit on whole dataset)
    cat_encoders = fit_label_encoders(data, ADULT_CATEGORICAL_COLUMNS)
    data = transform_with_label_encoders(data, cat_encoders)

    # Normalize numerical columns (fit on whole dataset)
    scaler = fit_standard_scaler(data, ADULT_NUMERICAL_COLUMNS_ALL)
    data = transform_with_scaler(data, scaler, ADULT_NUMERICAL_COLUMNS_ALL)

    X = data.drop("income", axis=1)
    y = fit_y_encoder(data["income"]).transform(data["income"])
    return X, y


def load_adult_age3(url, sensitive_feature):
    """
    Loads Adult dataset and partitions it into three age-based clients for federated learning.

    Args:
        url (str): Path or URL to the raw Adult dataset CSV.
        sensitive_feature (str): Column name used for fairness grouping (e.g., 'race' or 'sex').

    Returns
    -------
        data_dict: dict
            Dictionary mapping 'client_1' (age 0-29), 'client_2' (age 30-39), and 'client_3' (age 40+) to their respective PyTorch tensors {X, y, s, y_pot}.

        X_test: torch.Tensor
             A combined PyTorch tensor containing test features from all three age partitions.

        y_test: torch.Tensor 
            A combined PyTorch tensor containing binary labels for the test set.

        sex_list: list
            A list of raw sensitive feature values extracted from the combined test set.

        column_names_list: list
            A list of strings representing the feature column names.

        ytest_potential: torch.Tensor
            A PyTorch tensor containing potential outcome labels for the combined test set.
    
    """

    data = pd.read_csv(url)

    # Encode categoricals (global fit)
    cat_encoders = fit_label_encoders(data, ADULT_CATEGORICAL_COLUMNS)
    data = transform_with_label_encoders(data, cat_encoders)

    # Scale numerics without age (global fit)
    scaler = fit_standard_scaler(data, ADULT_NUMERICAL_COLUMNS_EXCL_AGE)
    data = transform_with_scaler(data, scaler, ADULT_NUMERICAL_COLUMNS_EXCL_AGE)

    # shuffle + dropna
    data = shuffle_dropna(data, random_state=DEFAULT_RANDOM_STATE)

    # Age partitions
    df1 = data[(data["age"] >= 0) & (data["age"] <= 29)].copy()
    df2 = data[(data["age"] >= 30) & (data["age"] <= 39)].copy()
    df3 = data[(data["age"] >= 40)].copy()

    # Scale age separately per group
    age_scaler = StandardScaler()
    for df in (df1, df2, df3):
        df["age"] = age_scaler.fit_transform(df[["age"]]).ravel().astype("float32")

    # Split each group into train/val/test
    df1_train, df1_val, df1_test = split_train_val_test(df1, random_state=DEFAULT_RANDOM_STATE)
    df2_train, df2_val, df2_test = split_train_val_test(df2, random_state=DEFAULT_RANDOM_STATE)
    df3_train, df3_val, df3_test = split_train_val_test(df3, random_state=DEFAULT_RANDOM_STATE)

    # Global y encoder (consistent across all splits)
    y_encoder = fit_y_encoder(data["income"])

    # Build client train dict
    train_dfs = [df1_train, df2_train, df3_train]
    client_names = ["client_1", "client_2", "client_3"]
    data_dict = make_federated_dict(
        client_names,
        train_dfs,
        target_col="income",
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    # Global Val
    X_val, y_val, sval_list, _val_feature_names, yval_potential = make_global_split_tensors(
        [df1_val, df2_val, df3_val],
        target_col="income",
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    # Global Test
    X_test, y_test, sex_list, column_names_list, ytest_potential = make_global_split_tensors(
        [df1_test, df2_test, df3_test],
        target_col="income",
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    return (
        data_dict,
        X_test, y_test, sex_list, column_names_list, ytest_potential,
        X_val, y_val, sval_list, yval_potential
    )


def load_adult_age5(url, sensitive_feature):
    """
    Loads Adult dataset and partitions it into five age-based clients for federated learning.

    Args:
        url (str): Path or URL to the raw Adult dataset CSV.
        sensitive_feature (str): Column name used for fairness grouping (e.g., 'race' or 'sex').

    Returns
    -------
        data_dict: dict
            Dictionary mapping client_1' through 'client_5' (stratified by age) to their respective PyTorch tensors {X, y, s, y_pot}.

        X_test: torch.Tensor
             A combined PyTorch tensor containing test features from all three age partitions.

        y_test: torch.Tensor 
            A combined PyTorch tensor containing binary labels for the test set.

        sex_list: list
            A list of raw sensitive feature values extracted from the combined test set.

        column_names_list: list
            A list of strings representing the feature column names.

        ytest_potential: torch.Tensor
            A PyTorch tensor containing potential outcome labels for the combined test set.
    
    """

    data = pd.read_csv(url)

    # Encode categoricals (global fit)
    cat_encoders = fit_label_encoders(data, ADULT_CATEGORICAL_COLUMNS)
    data = transform_with_label_encoders(data, cat_encoders)

    # Scale numerics without age (global fit)
    scaler = fit_standard_scaler(data, ADULT_NUMERICAL_COLUMNS_EXCL_AGE)
    data = transform_with_scaler(data, scaler, ADULT_NUMERICAL_COLUMNS_EXCL_AGE)

    # shuffle + dropna
    data = shuffle_dropna(data, random_state=DEFAULT_RANDOM_STATE)

    # Age partitions
    df1 = data[(data["age"] >= 0) & (data["age"] <= 30)].copy()
    df2 = data[(data["age"] >= 31) & (data["age"] <= 35)].copy()
    df3 = data[(data["age"] >= 36) & (data["age"] <= 45)].copy()
    df4 = data[(data["age"] >= 46) & (data["age"] <= 55)].copy()
    df5 = data[(data["age"] >= 56)].copy()

    # Scale age separately per group
    age_scaler = StandardScaler()
    for df in (df1, df2, df3, df4, df5):
        df["age"] = age_scaler.fit_transform(df[["age"]]).ravel().astype("float32")

    # Split each group into train/val/test
    splits = [split_train_val_test(df, random_state=DEFAULT_RANDOM_STATE) for df in (df1, df2, df3, df4, df5)]
    train_dfs = [s[0] for s in splits]
    val_dfs = [s[1] for s in splits]
    test_dfs = [s[2] for s in splits]

    # Global y encoder (consistent across all splits)
    y_encoder = fit_y_encoder(data["income"])

    # Build client train dict
    client_names = [f"client_{i}" for i in range(1, 6)]
    data_dict = make_federated_dict(
        client_names,
        train_dfs,
        target_col="income",
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    # Global Val
    X_val, y_val, sval_list, _val_feature_names, yval_potential = make_global_split_tensors(
        val_dfs,
        target_col="income",
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    # Global Test
    X_test, y_test, sex_list, column_names_list, ytest_potential = make_global_split_tensors(
        test_dfs,
        target_col="income",
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    return (
        data_dict,
        X_test, y_test, sex_list, column_names_list, ytest_potential,
        X_val, y_val, sval_list, yval_potential
    )


def load_adult_random(url, sensitive_feature, num_clients):
    """
    Loads Adult dataset and partitions it into N randomly split clients for federated learning.

    Args:
        url (str): Path or URL to the raw Adult dataset CSV.
        sensitive_feature (str): Column name used for fairness grouping (e.g., 'race' or 'sex').
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

    data = pd.read_csv(url)

    # Encode categoricals
    cat_encoders = fit_label_encoders(data, ADULT_CATEGORICAL_COLUMNS)
    data = transform_with_label_encoders(data, cat_encoders)

    # Scale numerics
    scaler = fit_standard_scaler(data, ADULT_NUMERICAL_COLUMNS_ALL)
    data = transform_with_scaler(data, scaler, ADULT_NUMERICAL_COLUMNS_ALL)

    data = shuffle_dropna(data, random_state=DEFAULT_RANDOM_STATE)

    # Split into N client chunks
    client_dfs = np.array_split(data, num_clients)

    # Global y encoder
    y_encoder = fit_y_encoder(data["income"])

    # per client: train/val/test
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
        target_col="income",
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    # Global val/test
    X_val, y_val, sval_list, _val_feature_names, yval_potential = make_global_split_tensors(
        val_dfs,
        target_col="income",
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )
    X_test, y_test, sex_list, column_names_list, ytest_potential = make_global_split_tensors(
        test_dfs,
        target_col="income",
        sensitive_feature=sensitive_feature,
        y_encoder=y_encoder,
        task="classification",
    )

    return (
        data_dict,
        X_test, y_test, sex_list, column_names_list, ytest_potential,
        X_val, y_val, sval_list, yval_potential
    )

if __name__ == "__main__":
    a, _, _, _, _, _ = load_adult_age5("../Datasets/adult.csv", "sex")
    print(a)
