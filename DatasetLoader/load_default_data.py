import torch
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from pathlib import Path


def load_default(url):
    """
    Loads encodes, and normalizes the Default Credit dataset (KDD)

    Steps:
    - Reads CSV data
    - Encodes selected categorical/ordinal attributes
    - Normalizes numerical attributes
    - Splits into features X and label y

    Args:
        url (str): Path to the dataset CSV file

    Returns:
    -------
        X (pd.DataFrame): Processed feature matrix with encoded categories and scaled numerical values.
        y (np.ndarray): Encoded target vector ('y') representing whether the client subscribed to a term deposit.
    """

    data = pd.read_csv(url)

    # Encode categorical / columns using Label Encoding
    categorical_columns = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
    ]
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize continuous numerical columns
    numerical_columns = [
        'AGE',
        'BILL_AMT1','BILL_AMT2','BILL_AMT3',
        'BILL_AMT4','BILL_AMT5','BILL_AMT6',
        'PAY_AMT1','PAY_AMT2','PAY_AMT3',
        'PAY_AMT4','PAY_AMT5','PAY_AMT6'
    ]
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Separate features and target label
    X = data.drop('y', axis=1)
    y = LabelEncoder().fit_transform(data['y'])

    return X, y


def load_default_age(url, sensitive_feature):
    """
    Loads the Default Credit dataset (KDD) and creates a NON-IID federated setup
    by splitting clients according to AGE groups.

    Args:
        url (str): Dataset path
        sensitive_feature (str): The column name to be used as the sensitive attribute.

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
    #data = shuffle(data)

    # Encode categorical columns
    categorical_columns = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical columns
    numerical_columns = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    data = shuffle(data)
    
    #train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    #data = train_df
    mask = (data['AGE'] >=0) & (data['AGE'] <=29)
    df1 = data[mask]
    mask = (data['AGE'] >=30) & (data['AGE'] <=39)
    df2 = data[mask]
    mask = (data['AGE'] >=40)
    df3 = data[mask]
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
   
    #scalrising the 'age' column as well
    numerical_columns = ['AGE']
    scaler = StandardScaler()
    df1[numerical_columns] = scaler.fit_transform(df1[numerical_columns])
    df2[numerical_columns] = scaler.fit_transform(df2[numerical_columns])
    df3[numerical_columns] = scaler.fit_transform(df3[numerical_columns])
    
    
    df1, test_df1 = train_test_split(df1, test_size=0.1, random_state=42)
    df2, test_df2 = train_test_split(df2, test_size=0.1, random_state=42)
    df3, test_df3 = train_test_split(df3, test_size=0.1, random_state=42)
    # Split the data into features and labels
    test_df = result = pd.concat([test_df1, test_df2, test_df3], ignore_index=True)
    test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
    X_client1 = df1.drop('y', axis=1)
    y_client1 = LabelEncoder().fit_transform(df1['y'])
    
    X_client2 = df2.drop('y', axis=1)
    y_client2 = LabelEncoder().fit_transform(df2['y'])
    
    X_client3 = df3.drop('y', axis=1)
    y_client3 = LabelEncoder().fit_transform(df3['y'])
    
    
    
    s_client1 = X_client1[sensitive_feature]
    #y_potential_client1 = find_potential_outcomes(X_client1,y_client1, sensitive_feature)
    y_potential_client1 =y_client1 
    X_client1 = torch.tensor(X_client1.values, dtype=torch.float32)
    y_client1 = torch.tensor(y_client1, dtype=torch.float32)
    y_potential_client1 =y_client1 
    s_client1 = torch.from_numpy(s_client1.values).float()
    y_potential_client1 = torch.tensor(y_potential_client1, dtype=torch.float32)
   
    
    
    
    
    s_client2 = X_client2[sensitive_feature]
    #y_potential_client2 = find_potential_outcomes(X_client2,y_client2, sensitive_feature)
    y_potential_client2 =y_client2 
    X_client2 = torch.tensor(X_client2.values, dtype=torch.float32)
    y_client2 = torch.tensor(y_client2, dtype=torch.float32)
    s_client2 = torch.from_numpy(s_client2.values).float()
    y_potential_client2 = torch.tensor(y_potential_client2, dtype=torch.float32)
    
    
    s_client3 = X_client3[sensitive_feature]
    #y_potential_client3 = find_potential_outcomes(X_client3,y_client3, sensitive_feature)
    y_potential_client3 =y_client3 
    X_client3 = torch.tensor(X_client3.values, dtype=torch.float32)
    y_client3 = torch.tensor(y_client3, dtype=torch.float32)
    s_client3 = torch.from_numpy(s_client3.values).float()
    y_potential_client3 = torch.tensor(y_potential_client3, dtype=torch.float32)
    
    
    
    X_test = test_df.drop('y', axis=1)
   
    y_test = LabelEncoder().fit_transform(test_df['y'])
    sex_column = X_test[sensitive_feature]
    column_names_list = X_test.columns.tolist()
    #ytest_potential = find_potential_outcomes(X_test,y_test, sensitive_feature)
    ytest_potential =y_test 
    ytest_potential = torch.tensor(ytest_potential, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    
    sex_list = sex_column.tolist()
    data_dict = {}
    data_dict["client_1"] = {"X": X_client1, "y": y_client1, "s": s_client1, "y_pot": y_potential_client1}
    data_dict["client_2"] = {"X": X_client2, "y": y_client2, "s": s_client2, "y_pot": y_potential_client2}
    data_dict["client_3"] = {"X": X_client3, "y": y_client3, "s": s_client3, "y_pot": y_potential_client3}
    
    return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential

def load_default_random(url, sensitive_feature, num_clients):
    """
    Loads Default Credit dataset (KDD) and partitions it into N randomly split clients for federated learning.

    Args:
        url (str): Path or URL to the Default Credit dataset (KDD) CSV.
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

    # 1. Load dataset
    data = pd.read_csv(url)

    # 2. Encode categorical / ordinal columns
    categorical_columns = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
    ]
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # 3. Normalize ALL numerical columns
    numerical_columns = [
        'AGE',
        'BILL_AMT1','BILL_AMT2','BILL_AMT3',
        'BILL_AMT4','BILL_AMT5','BILL_AMT6',
        'PAY_AMT1','PAY_AMT2','PAY_AMT3',
        'PAY_AMT4','PAY_AMT5','PAY_AMT6'
    ]
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # 4. Shuffle dataset
    data = shuffle(data, random_state=42)

    # 5. Split data into N client chunks
    client_dfs = np.array_split(data, num_clients)

    data_dict = {}
    test_dfs_list = []

    # 6. Process each client chunk
    for i, df_chunk in enumerate(client_dfs):
        client_name = f"client_{i+1}"

        df_chunk = df_chunk.dropna()

        df_train, df_test = train_test_split(
            df_chunk, test_size=0.1, random_state=42
        )

        test_dfs_list.append(df_test)

        X_client = df_train.drop('y', axis=1)
        y_client = LabelEncoder().fit_transform(df_train['y'])

        s_client = X_client[sensitive_feature]

        y_potential_client = y_client

        X_tensor = torch.tensor(X_client.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_client, dtype=torch.float32)
        s_tensor = torch.from_numpy(s_client.values).float()
        y_pot_tensor = torch.tensor(y_potential_client, dtype=torch.float32)

        data_dict[client_name] = {
            "X": X_tensor,
            "y": y_tensor,
            "s": s_tensor,
            "y_pot": y_pot_tensor
        }

    # 7. Process Global Test Set
    test_df = pd.concat(test_dfs_list, ignore_index=True)

    X_test = test_df.drop('y', axis=1)
    y_test = LabelEncoder().fit_transform(test_df['y'])

    sex_column = X_test[sensitive_feature]
    column_names_list = X_test.columns.tolist()
    sex_list = sex_column.tolist()

    ytest_potential = y_test
    ytest_potential_tensor = torch.tensor(ytest_potential, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
   

    return (data_dict, X_test_tensor, y_test_tensor, sex_list, column_names_list, ytest_potential_tensor)

if __name__ == "__main__":
    a, _,_,_,_,_ = load_default_random("../Datasets/default.csv", "AGE",7)
    print(a)