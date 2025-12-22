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
from collections import defaultdict

STATES_LIST = [
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga',
    'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md',
    'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj',
    'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc',
    'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy'
]


def loadAndPreprocessACS():
    """
    Internal helper:
    1. Loads all 50 state CSVs.
    2. Merges them into one DataFrame (keeping track of the 'State' source).
    3. Encodes Categorical variables (globally to ensure consistency).
    4. Normalizes Numerical variables (globally).
    """
    dfs = []
    for state in STATES_LIST:
        try:
            df = pd.read_csv(f'../Datasets/acs_dataset/{state}_data.csv')
            # Track state for splitting later
            df['STATE_ID'] = state 
            dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: File for {state} not found. Skipping.")
            continue
            
    if not dfs:
        raise ValueError("No data files found.")

    data = pd.concat(dfs, ignore_index=True)
    
    # 1. Pre-process Sensitive Attribute (RAC1P)
    #    Target: 1 if White (1), 0 otherwise
    if 'RAC1P' in data.columns:
        data['RAC1P'] = data['RAC1P'].apply(lambda x: 1 if x == 1 else 0)

    # 2. Encode Categorical Columns
    #    Note: 'AGEP' is numerical here, unlike original Adult dataset
    categorical_columns = ['AGEP', 'COW', 'SCHL', 'MAR', 'RELP', 'SEX', 'STATE_ID'] 
    numerical_columns = ['OCCP', 'POBP', 'WKHP']
    
    # We fit encoders on the WHOLE dataset so 'CA' is always encoded to the same int
    for col in categorical_columns:
        if col in data.columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])

    # 3. Normalize Numerical Columns
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    return data, numerical_columns, scaler

def load_acs():
    """
    Loads and processes ACS Income data from state-level CSVs into a federated PyTorch format.

    Inputs:
        None (Reads .csv files from the './acs_dataset/' directory).

    Returns
    -------
    data_dict : dict
        Maps 'client_i' to a dict of torch.Tensor objects {X, y, s, y_pot} per state.
    X_test : torch.Tensor
        Combined and normalized test features from all states.
    y_test : torch.Tensor
        Combined binary income labels (1 if >$50k, 0 otherwise).
    sex_list : list
        Sensitive feature values (RAC1P) for the combined test set.
    column_names_list : list
        List of feature strings included in the tensors.
    y_pot : torch.Tensor
        A zero-filled placeholder tensor for potential outcome logic.
    """
    #states = ['ca', 'il', 'ny', 'tx', 'fl', 'pa', 'oh', 'mi', 'ga', 'nc'] #10 states
     #all states
    states = [
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga',
    'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md',
    'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj',
    'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc',
    'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy'
    ]
    
    categorical_columns = ['AGEP', 'COW', 'SCHL', 'MAR', 'RELP', 'SEX']
    numerical_columns = ['OCCP', 'POBP', 'WKHP']
    sensitive_feature = 'RAC1P'
    data_dict = defaultdict(list)
    scaler = StandardScaler()
    test_dfs = defaultdict(list)
    client_num = 1
    for state in states:
        data = pd.read_csv(f'../Datasets/acs_dataset/{state}_data.csv')
        data['RAC1P'] = data['RAC1P'].apply(lambda x: 1 if x == 1 else 0)
        for col in categorical_columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])
        # Normalize numerical columns
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        data, test_df = train_test_split(data, test_size=0.1, random_state=42)
        client_name = "client_"+str(client_num)
        test_dfs[client_name]=test_df
        
        # Split the data into features and labels
        X_client = data.drop('PINCP', axis=1)
        y_client = LabelEncoder().fit_transform(data['PINCP'])
        s_client = X_client[sensitive_feature]
        #y_potential_client1 = find_potential_outcomes(X_client1,y_client1, sensitive_feature)
        X_client = torch.tensor(X_client.values, dtype=torch.float32)
        y_client = torch.tensor(y_client, dtype=torch.float32)
        s_client = torch.from_numpy(s_client.values).float()
        #y_potential_client1 = torch.tensor(y_potential_client1, dtype=torch.float32)
        y_pot = torch.zeros_like(y_client)
        update_data = {"X": X_client, "y": y_client, "s": s_client, "y_pot": y_pot}
        data_dict[client_name]=update_data
        client_num +=1
    
    # Concatenate the dataframes into a single dataframe
    test_df = pd.concat(test_dfs.values(), ignore_index=True)
    test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
    X_test = test_df.drop('PINCP', axis=1)
    y_test = LabelEncoder().fit_transform(test_df['PINCP'])
    sex_column = X_test[sensitive_feature]
    sex_list = sex_column.tolist()
    column_names_list = X_test.columns.tolist()
    #ytest_potential = find_potential_outcomes(X_test,y_test, sensitive_feature)
    #ytest_potential = torch.tensor(ytest_potential, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    y_pot = torch.zeros_like(y_test)
    
    return data_dict, X_test, y_test, sex_list, column_names_list,y_pot

def load_acs_states_3(sensitive_feature='RAC1P'):
    """
    Loads and processes ACS Income data from state-level CSVs into a federated PyTorch format.
    The data is split for 3 clients splitting the 50 states across 3 clients randomly.
    Data for each state is not split, meaning every state is specific to one client

    Returns
    -------
    data_dict : dict
        Maps 'client_i' to a dict of torch.Tensor objects {X, y, s, y_pot} per state.
    X_test : torch.Tensor
        Combined and normalized test features from all states.
    y_test : torch.Tensor
        Combined binary income labels (1 if >$50k, 0 otherwise).
    sex_list : list
        Sensitive feature values (RAC1P) for the combined test set.
    column_names_list : list
        List of feature strings included in the tensors.
    y_pot : torch.Tensor
        A zero-filled placeholder tensor for potential outcome logic.
    """
    data, numerical_columns, scaler = loadAndPreprocessACS()
    
    unique_states = data['STATE_ID'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_states)
    
    state_groups = np.array_split(unique_states, 3)
    
    df1 = data[data['STATE_ID'].isin(state_groups[0])].copy()
    df2 = data[data['STATE_ID'].isin(state_groups[1])].copy()
    df3 = data[data['STATE_ID'].isin(state_groups[2])].copy()
    
    df1, test_df1 = train_test_split(df1, test_size=0.1, random_state=42)
    df2, test_df2 = train_test_split(df2, test_size=0.1, random_state=42)
    df3, test_df3 = train_test_split(df3, test_size=0.1, random_state=42)

    test_df = pd.concat([test_df1, test_df2, test_df3], ignore_index=True)
    
    test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
    
    clients = [df1, df2, df3]
    data_dict = {}
    
    for i, df in enumerate(clients):
        client_name = f"client_{i+1}"
        df_clean = df.drop(['PINCP', 'STATE_ID'], axis=1)
        
        y_cl = LabelEncoder().fit_transform(df['PINCP'])
        s_cl = df_clean[sensitive_feature]
        
        X_tensor = torch.tensor(df_clean.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_cl, dtype=torch.float32)
        s_tensor = torch.from_numpy(s_cl.values).float()
        y_pot = torch.zeros_like(y_tensor)
        
        data_dict[client_name] = {"X": X_tensor, "y": y_tensor, "s": s_tensor, "y_pot": y_pot}

    X_test = test_df.drop(['PINCP', 'STATE_ID'], axis=1)
    y_test = LabelEncoder().fit_transform(test_df['PINCP'])
    sex_list = X_test[sensitive_feature].tolist()
    col_names = X_test.columns.tolist()
    
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_pot_test = torch.zeros_like(y_test_tensor)
    
    return data_dict, X_test_tensor, y_test_tensor, sex_list, col_names, y_pot_test


def load_acs_states_3(sensitive_feature='RAC1P'):
    """
    Loads and processes ACS Income data from state-level CSVs into a federated PyTorch format.
    The data is split for 5 clients splitting the 50 states across 5 clients randomly.
    Data for each state is not split, meaning every state is specific to one client

    Returns
    -------
    data_dict : dict
        Maps 'client_i' to a dict of torch.Tensor objects {X, y, s, y_pot} per state.
    X_test : torch.Tensor
        Combined and normalized test features from all states.
    y_test : torch.Tensor
        Combined binary income labels (1 if >$50k, 0 otherwise).
    sex_list : list
        Sensitive feature values (RAC1P) for the combined test set.
    column_names_list : list
        List of feature strings included in the tensors.
    y_pot : torch.Tensor
        A zero-filled placeholder tensor for potential outcome logic.
    """
    data, numerical_columns, scaler = loadAndPreprocessACS()
    
    unique_states = data['STATE_ID'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_states)
    
    state_groups = np.array_split(unique_states, 5)
    
    df1 = data[data['STATE_ID'].isin(state_groups[0])].copy()
    df2 = data[data['STATE_ID'].isin(state_groups[1])].copy()
    df3 = data[data['STATE_ID'].isin(state_groups[2])].copy()
    df4 = data[data['STATE_ID'].isin(state_groups[3])].copy()
    df5 = data[data['STATE_ID'].isin(state_groups[4])].copy()
    
    
    
    df1, test_df1 = train_test_split(df1, test_size=0.1, random_state=42)
    df2, test_df2 = train_test_split(df2, test_size=0.1, random_state=42)
    df3, test_df3 = train_test_split(df3, test_size=0.1, random_state=42)
    df4, test_df4 = train_test_split(df4, test_size=0.1, random_state=42)
    df5, test_df5 = train_test_split(df5, test_size=0.1, random_state=42)

    test_df = pd.concat([test_df1, test_df2, test_df3, test_df4, test_df5], ignore_index=True)
    
    test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
    
    clients = [df1, df2, df3,df4,df5]
    data_dict = {}
    
    for i, df in enumerate(clients):
        client_name = f"client_{i+1}"
        df_clean = df.drop(['PINCP', 'STATE_ID'], axis=1)
        
        y_cl = LabelEncoder().fit_transform(df['PINCP'])
        s_cl = df_clean[sensitive_feature]
        
        X_tensor = torch.tensor(df_clean.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_cl, dtype=torch.float32)
        s_tensor = torch.from_numpy(s_cl.values).float()
        y_pot = torch.zeros_like(y_tensor)
        
        data_dict[client_name] = {"X": X_tensor, "y": y_tensor, "s": s_tensor, "y_pot": y_pot}

    X_test = test_df.drop(['PINCP', 'STATE_ID'], axis=1)
    y_test = LabelEncoder().fit_transform(test_df['PINCP'])
    sex_list = X_test[sensitive_feature].tolist()
    col_names = X_test.columns.tolist()
    
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_pot_test = torch.zeros_like(y_test_tensor)
    
    return data_dict, X_test_tensor, y_test_tensor, sex_list, col_names, y_pot_test

def load_acs_random(sensitive_feature='RAC1P', num_clients=10):
    """
    Loads and processes ACS Income data from state-level CSVs into a federated PyTorch format.
    The data is split for N clients in an IID way by randomly splitting the data while breaking up the
    structure of the states.

    Args:
        url (str): Path or URL to the raw Adult dataset CSV.
        sensitive_feature (str): Column name used for fairness grouping (e.g., 'race' or 'sex').

    Returns
    -------
    data_dict : dict
        Maps 'client_i' to a dict of torch.Tensor objects {X, y, s, y_pot} per state.
    X_test : torch.Tensor
        Combined and normalized test features from all states.
    y_test : torch.Tensor
        Combined binary income labels (1 if >$50k, 0 otherwise).
    sex_list : list
        Sensitive feature values (RAC1P) for the combined test set.
    column_names_list : list
        List of feature strings included in the tensors.
    y_pot : torch.Tensor
        A zero-filled placeholder tensor for potential outcome logic.
    """
        
    data, numerical_columns, scaler = loadAndPreprocessACS()
    
    data = data.drop('STATE_ID', axis=1)
    
    data = shuffle(data, random_state=42)
    
    client_dfs = np.array_split(data, num_clients)
    
    data_dict = {}
    test_dfs_list = []
    
    for i, df_chunk in enumerate(client_dfs):
        client_name = f"client_{i+1}"
        
        df_train, df_test = train_test_split(df_chunk, test_size=0.1, random_state=42)
        test_dfs_list.append(df_test)
        
        X_cl = df_train.drop('PINCP', axis=1)
        y_cl = LabelEncoder().fit_transform(df_train['PINCP'])
        s_cl = X_cl[sensitive_feature]
        
        X_tensor = torch.tensor(X_cl.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_cl, dtype=torch.float32)
        s_tensor = torch.from_numpy(s_cl.values).float()
        y_pot = torch.zeros_like(y_tensor)
        
        data_dict[client_name] = {"X": X_tensor, "y": y_tensor, "s": s_tensor, "y_pot": y_pot}
        
    test_df = pd.concat(test_dfs_list, ignore_index=True)
    test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
    
    X_test = test_df.drop('PINCP', axis=1)
    y_test = LabelEncoder().fit_transform(test_df['PINCP'])
    sex_list = X_test[sensitive_feature].tolist()
    col_names = X_test.columns.tolist()
    
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_pot_test = torch.zeros_like(y_test_tensor)
    
    return data_dict, X_test_tensor, y_test_tensor, sex_list, col_names, y_pot_test

if __name__ == "__main__":
    print("Testing load_acs_states_3...")
    data_dict, X_test, y_test, _, _, _ = load_acs_states_3()
    print(f"Client 1 (State Group 1) Size: {data_dict['client_1']['X'].shape}")
    print(f"Client 2 (State Group 2) Size: {data_dict['client_2']['X'].shape}")
    print(f"Client 3 (State Group 3) Size: {data_dict['client_3']['X'].shape}")
    print(f"Global Test Set Size: {X_test.shape}")

    