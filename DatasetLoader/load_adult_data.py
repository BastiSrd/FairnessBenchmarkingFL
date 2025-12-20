import torch
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
from sklearn.neighbors import NearestNeighbors

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
    #data = shuffle(data)
   

    # Encode categorical columns
    categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
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


def load_adult_age(url, sensitive_feature):
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
    #data = shuffle(data)

    # Encode categorical columns
    categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        
    # Normalize numerical columns
    numerical_columns = ['fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    data = shuffle(data)
    
    #train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    #data = train_df
    mask = (data['age'] >=0) & (data['age'] <=29)
    df1 = data[mask]
    mask = (data['age'] >=30) & (data['age'] <=39)
    df2 = data[mask]
    mask = (data['age'] >=40)
    df3 = data[mask]
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
   
    #scalrising the 'age' column as well
    numerical_columns = ['age']
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
    X_client1 = df1.drop('income', axis=1)
    y_client1 = LabelEncoder().fit_transform(df1['income'])
    
    X_client2 = df2.drop('income', axis=1)
    y_client2 = LabelEncoder().fit_transform(df2['income'])
    
    X_client3 = df3.drop('income', axis=1)
    y_client3 = LabelEncoder().fit_transform(df3['income'])
    
    
    
    s_client1 = X_client1[sensitive_feature]
    #y_potential_client1 = find_potential_outcomes(X_client1,y_client1, sensitive_feature)
    y_potential_client1 =y_client1 
    X_client1 = torch.tensor(X_client1.values, dtype=torch.float32)
    y_client1 = torch.tensor(y_client1, dtype=torch.float32)
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
    
    
    
    X_test = test_df.drop('income', axis=1)
   
    y_test = LabelEncoder().fit_transform(test_df['income'])
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
    #print("bismillah")
    
    return data_dict, X_test, y_test, sex_list, column_names_list,ytest_potential

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

    # 1. Encode categorical columns
    categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        
    # 2. Normalize standard numerical columns (excluding 'age' initially, matching original logic)
    numerical_columns = ['fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    # 3. Shuffle data before splitting to ensure random IID partitions
    data = shuffle(data, random_state=42)
    
    # 4. Split data into N chunks
    client_dfs = np.array_split(data, num_clients)
    
    data_dict = {}
    test_dfs_list = []
    
    # Define age column for specific scaling per client (matching original logic)
    age_col = ['age']
    
    # 5. Process each client chunk
    for i, df_chunk in enumerate(client_dfs):
        client_name = f"client_{i+1}"
        
        # Drop NAs
        df_chunk = df_chunk.dropna()
        
        # Scale 'age' column specifically for this client partition
        scaler_age = StandardScaler()
        df_chunk[age_col] = scaler_age.fit_transform(df_chunk[age_col])
        
        # Split train/test (90/10)
        df_train, df_test = train_test_split(df_chunk, test_size=0.1, random_state=42)
        
        # Save test set for later aggregation
        test_dfs_list.append(df_test)
        
        # Prepare Train Features and Target
        # Note: 'income' is the target in Adult dataset
        X_client = df_train.drop('income', axis=1)
        y_client = LabelEncoder().fit_transform(df_train['income'])
        
        # Extract sensitive feature
        s_client = X_client[sensitive_feature]
        
        # Placeholder for potential outcomes (matching original logic)
        y_potential_client = y_client
        
        # Convert to Tensors
        X_tensor = torch.tensor(X_client.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_client, dtype=torch.float32)
        s_tensor = torch.from_numpy(s_client.values).float()
        y_pot_tensor = torch.tensor(y_potential_client, dtype=torch.float32)
        
        # Store in dictionary
        data_dict[client_name] = {
            "X": X_tensor, 
            "y": y_tensor, 
            "s": s_tensor, 
            "y_pot": y_pot_tensor
        }

    # 6. Process Global Test Set
    test_df = pd.concat(test_dfs_list, ignore_index=True)
    
    # Re-normalize 'age' on the combined test set
    scaler_test = StandardScaler()
    test_df[age_col] = scaler_test.fit_transform(test_df[age_col])
    
    X_test = test_df.drop('income', axis=1)
    y_test = LabelEncoder().fit_transform(test_df['income'])
    
    # Extract Metadata
    sex_column = X_test[sensitive_feature]
    column_names_list = X_test.columns.tolist()
    sex_list = sex_column.tolist()
    
    # Prepare Test Tensors
    ytest_potential = y_test
    ytest_potential_tensor = torch.tensor(ytest_potential, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    return data_dict, X_test_tensor, y_test_tensor, sex_list, column_names_list, ytest_potential_tensor


if __name__ == "__main__":
    a, _,_,_,_,_ = load_adult_random("../Datasets/adult.csv", "sex",7)
    print(a)

