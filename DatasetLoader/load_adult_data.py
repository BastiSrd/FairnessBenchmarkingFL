import torch
from psmpy.plotting import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle


def split_train_val_test(df, random_state=42):
    """
    Helper function to split a dataframe into train, validation, and test sets (70/15/15).

    :param df: Dataframe to split
    :param random_state: Random state to use
    :return: Train, validation and test sets
    """
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=random_state)
    return train_df, val_df, test_df


def client_tensors(df_split, sensitive_feature, y_encoder=None):
    """
    Convert a split dataframe into tensors for FL.


    :param df_split: Must contain 'income' + feature columns.
    :param sensitive_feature: Column name inside X used as sensitive attribute.
    :param y_encoder: If provided, used to transform income consistently.

    :return X_t (torch.FloatTensor): shape (N, D)
    :return y_t (torch.FloatTensor): shape (N,)
    :return s_t (torch.FloatTensor): shape (N,)
    :return y_pot_t (torch.FloatTensor): shape (N,) (currently same as y)
    """
    X = df_split.drop('income', axis=1)

    if y_encoder is None:
        y = LabelEncoder().fit_transform(df_split['income'])
    else:
        y = y_encoder.transform(df_split['income'])

    s = X[sensitive_feature].to_numpy()
    y_pot = y

    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    s_t = torch.tensor(s, dtype=torch.float32)
    y_pot_t = torch.tensor(y_pot, dtype=torch.float32)

    return X_t, y_t, s_t, y_pot_t


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

    # Encode categorical columns
    categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex',
                           'native.country']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    # Split the data into features and labels
    X = data.drop('income', axis=1)
    y = LabelEncoder().fit_transform(data['income'])

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

    categorical_columns = [
        'workclass', 'education', 'marital.status', 'occupation',
        'relationship', 'race', 'sex', 'native.country'
    ]
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    numerical_columns = ['fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    data = shuffle(data, random_state=42).dropna()

    df1 = data[(data['age'] >= 0) & (data['age'] <= 29)].copy()
    df2 = data[(data['age'] >= 30) & (data['age'] <= 39)].copy()
    df3 = data[(data['age'] >= 40)].copy()

    age_scaler = StandardScaler()
    for df in (df1, df2, df3):
        df['age'] = age_scaler.fit_transform(df[['age']]).ravel().astype('float32')

    df1_train, df1_val, df1_test = split_train_val_test(df1)
    df2_train, df2_val, df2_test = split_train_val_test(df2)
    df3_train, df3_val, df3_test = split_train_val_test(df3)

    # Optional: consistent income encoding across all splits
    y_encoder = LabelEncoder()
    y_encoder.fit(data['income'])

    # Train tensors
    X1, y1, s1, ypot1 = client_tensors(df1_train, sensitive_feature, y_encoder)
    X2, y2, s2, ypot2 = client_tensors(df2_train, sensitive_feature, y_encoder)
    X3, y3, s3, ypot3 = client_tensors(df3_train, sensitive_feature, y_encoder)

    # Global Val
    val_df = pd.concat([df1_val, df2_val, df3_val], ignore_index=True)
    X_val_df = val_df.drop('income', axis=1)
    y_val_np = y_encoder.transform(val_df['income'])

    s_val_column = X_val_df[sensitive_feature]
    X_val = torch.tensor(X_val_df.values, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    sval_list = s_val_column.tolist()
    yval_potential = torch.tensor(y_val_np, dtype=torch.float32)

    # Global Test
    test_df = pd.concat([df1_test, df2_test, df3_test], ignore_index=True)
    X_test_df = test_df.drop('income', axis=1)
    y_test_np = y_encoder.transform(test_df['income'])

    sex_column = X_test_df[sensitive_feature]
    column_names_list = X_test_df.columns.tolist()

    X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    sex_list = sex_column.tolist()
    ytest_potential = torch.tensor(y_test_np, dtype=torch.float32)

    data_dict = {
        "client_1": {"X": X1, "y": y1, "s": s1, "y_pot": ypot1},
        "client_2": {"X": X2, "y": y2, "s": s2, "y_pot": ypot2},
        "client_3": {"X": X3, "y": y3, "s": s3, "y_pot": ypot3},
    }

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

    categorical_columns = [
        'workclass', 'education', 'marital.status', 'occupation',
        'relationship', 'race', 'sex', 'native.country'
    ]
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    numerical_columns = ['fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    data = shuffle(data, random_state=42).dropna()

    df1 = data[(data['age'] >= 0) & (data['age'] <= 30)].copy()
    df2 = data[(data['age'] >= 31) & (data['age'] <= 35)].copy()
    df3 = data[(data['age'] >= 36) & (data['age'] <= 45)].copy()
    df4 = data[(data['age'] >= 46) & (data['age'] <= 55)].copy()
    df5 = data[(data['age'] >= 56)].copy()

    age_scaler = StandardScaler()
    for df in (df1, df2, df3, df4, df5):
        df['age'] = age_scaler.fit_transform(df[['age']]).ravel().astype('float32')

    df1_train, df1_val, df1_test = split_train_val_test(df1)
    df2_train, df2_val, df2_test = split_train_val_test(df2)
    df3_train, df3_val, df3_test = split_train_val_test(df3)
    df4_train, df4_val, df4_test = split_train_val_test(df4)
    df5_train, df5_val, df5_test = split_train_val_test(df5)

    # Consistent income encoding across all splits
    y_encoder = LabelEncoder()
    y_encoder.fit(data['income'])

    # Train tensors
    X1, y1, s1, ypot1 = client_tensors(df1_train, sensitive_feature, y_encoder)
    X2, y2, s2, ypot2 = client_tensors(df2_train, sensitive_feature, y_encoder)
    X3, y3, s3, ypot3 = client_tensors(df3_train, sensitive_feature, y_encoder)
    X4, y4, s4, ypot4 = client_tensors(df4_train, sensitive_feature, y_encoder)
    X5, y5, s5, ypot5 = client_tensors(df5_train, sensitive_feature, y_encoder)

    # Global Val
    val_df = pd.concat([df1_val, df2_val, df3_val, df4_val, df5_val], ignore_index=True)
    X_val_df = val_df.drop('income', axis=1)
    y_val_np = y_encoder.transform(val_df['income'])

    s_val_column = X_val_df[sensitive_feature]
    X_val = torch.tensor(X_val_df.values, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    sval_list = s_val_column.tolist()
    yval_potential = torch.tensor(y_val_np, dtype=torch.float32)

    # Global Test
    test_df = pd.concat([df1_test, df2_test, df3_test, df4_test, df5_test], ignore_index=True)
    X_test_df = test_df.drop('income', axis=1)
    y_test_np = y_encoder.transform(test_df['income'])

    sex_column = X_test_df[sensitive_feature]
    column_names_list = X_test_df.columns.tolist()

    X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    sex_list = sex_column.tolist()
    ytest_potential = torch.tensor(y_test_np, dtype=torch.float32)

    data_dict = {
        "client_1": {"X": X1, "y": y1, "s": s1, "y_pot": ypot1},
        "client_2": {"X": X2, "y": y2, "s": s2, "y_pot": ypot2},
        "client_3": {"X": X3, "y": y3, "s": s3, "y_pot": ypot3},
        "client_4": {"X": X4, "y": y4, "s": s4, "y_pot": ypot4},
        "client_5": {"X": X5, "y": y5, "s": s5, "y_pot": ypot5},
    }

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

    categorical_columns = [
        'workclass', 'education', 'marital.status', 'occupation',
        'relationship', 'race', 'sex', 'native.country'
    ]
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    numerical_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    data = shuffle(data, random_state=42).dropna()

    client_dfs = np.array_split(data, num_clients)

    # Consistent income encoding across all splits
    y_encoder = LabelEncoder()
    y_encoder.fit(data['income'])

    data_dict = {}
    val_dfs = []
    test_dfs = []

    for i, df_chunk in enumerate(client_dfs, start=1):
        df_train, df_val, df_test = split_train_val_test(df_chunk)

        # Train tensors only
        Xtr, ytr, str_, ypottr = client_tensors(df_train, sensitive_feature, y_encoder)
        data_dict[f"client_{i}"] = {"X": Xtr, "y": ytr, "s": str_, "y_pot": ypottr}

        val_dfs.append(df_val)
        test_dfs.append(df_test)

    # Global Val
    val_df = pd.concat(val_dfs, ignore_index=True)
    X_val_df = val_df.drop('income', axis=1)
    y_val_np = y_encoder.transform(val_df['income'])

    s_val_column = X_val_df[sensitive_feature]
    X_val = torch.tensor(X_val_df.values, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    sval_list = s_val_column.tolist()
    yval_potential = torch.tensor(y_val_np, dtype=torch.float32)

    # Global Test
    test_df = pd.concat(test_dfs, ignore_index=True)
    X_test_df = test_df.drop('income', axis=1)
    y_test_np = y_encoder.transform(test_df['income'])

    sex_column = X_test_df[sensitive_feature]
    column_names_list = X_test_df.columns.tolist()

    X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    sex_list = sex_column.tolist()
    ytest_potential = torch.tensor(y_test_np, dtype=torch.float32)

    return (
        data_dict,
        X_test, y_test, sex_list, column_names_list, ytest_potential,
        X_val, y_val, sval_list, yval_potential
    )


if __name__ == "__main__":
    a, _, _, _, _, _ = load_adult_age5("../Datasets/adult.csv", "sex")
    print(a)
