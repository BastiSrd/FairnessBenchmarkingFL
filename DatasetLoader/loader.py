import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def split_70_15_15(df, *, seed=42):
    """
    Splits a DataFrame into 70% train, 15% validation, 15% test. Default random_state of 42.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to split.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    train_df : pandas.DataFrame
        70% training split.
        15% validation split.
        15% test split.
    """
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed)
    return train_df, val_df, test_df


def dataframe_to_tensors(df, *, target_col, sensitive_feature, y_encoder):
    """
    Converts a DataFrame split into tensors used by FL clients.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame (typically a TRAIN split).
    target_col : str
        Name of the target column.
    sensitive_feature : str
        Name of the sensitive attribute column.
    y_encoder : sklearn.preprocessing.LabelEncoder
        Encoder fitted on the full dataset target column.

    Returns
    -------
    X : torch.FloatTensor
        Feature tensor.
        Target tensor.
        Sensitive attribute tensor.
        Potential outcome tensor (clone of y).
    """

    X_df = df.drop(columns=[target_col])

    y_np = y_encoder.transform(df[target_col]).astype(np.float32, copy=False)
    s_np = X_df[sensitive_feature].to_numpy(dtype=np.float32, copy=False)

    X = torch.tensor(X_df.values, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    s = torch.tensor(s_np, dtype=torch.float32)
    y_pot = y.clone()

    return X, y, s, y_pot

def encode_categoricals(df, categorical_cols):
    """
    Label-encodes the given categorical columns.
    """
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


def scale_numeric_cols(df, numeric_cols):
    """
    Standard-scales the given numeric columns.
    """
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def build_global_eval_sets(val_df, test_df, *, target_col, sensitive_feature, y_encoder):
    """
    Builds global validation and test tensors shared across all clients.

    Parameters
    ----------
    val_df : pandas.DataFrame
        Concatenated validation DataFrame from all clients.
    test_df : pandas.DataFrame
        Concatenated test DataFrame from all clients.
    target_col : str
        Name of the target column.
    sensitive_feature : str
        Name of the sensitive attribute column.
    y_encoder : sklearn.preprocessing.LabelEncoder
        Encoder fitted on the full dataset target column.

    Returns
    -------
    X_test : torch.FloatTensor
        Test feature tensor.
        Test target tensor.
        Sensitive attribute values for the test set (list format).
        Feature column names.
        Potential outcome labels for test set.
        Validation feature tensor.
        Validation target tensor.
        Sensitive attribute values for the validation set.
        Potential outcome labels for validation set.
    """
    X_val_df = val_df.drop(columns=[target_col])
    X_test_df = test_df.drop(columns=[target_col])

    X_val = torch.tensor(X_val_df.values, dtype=torch.float32)
    y_val = torch.tensor(
        y_encoder.transform(val_df[target_col]).astype("float32"),
        dtype=torch.float32,
    )
    s_val_list = X_val_df[sensitive_feature].tolist()
    y_val_pot = y_val.clone()

    X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_test = torch.tensor(
        y_encoder.transform(test_df[target_col]).astype("float32"),
        dtype=torch.float32,
    )
    s_test_list = X_test_df[sensitive_feature].tolist()
    col_names = X_test_df.columns.tolist()
    y_test_pot = y_test.clone()

    return (
        X_test, y_test, s_test_list, col_names, y_test_pot,
        X_val, y_val, s_val_list, y_val_pot
    )
