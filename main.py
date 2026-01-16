import torch
import numpy as np
import DatasetLoader
from DatasetLoader import load_adult_data, load_acs_data, load_bank_data
from DatasetLoader.load_kdd_data import load_kdd, load_kdd_random
from client import FLClient
from server import FLServer
import lossStrategies

# --- Configuration ---
ALGORITHM = 'FedAvg'  # Options: 'FedAvg', 'FedMinMax', 'TrustFed', 'Fairness' / only FedAvg currently implemented
LOADER = '3_clients'     # Options: '3_clients', '5_clients', 'random'
ROUNDS = 10
CLIENT_EPOCHS = 3
LR = 0.01

def runFLSimulation():
    print(f"--- Starting FL Simulation: {ALGORITHM} on {LOADER} ---")
    
    #Load Data
    print("Loading data...")
    if LOADER == '3_clients':
        data_dict, X_test, y_test, s_list, cols, ypot, X_val, y_val, sval_list, yvalpot = load_adult_data.load_adult_random() # replace function if other Dataset wanted
    elif LOADER == '5_clients':
        data_dict, X_test, y_test, s_list, _, _ = DatasetLoader.load_adult_data.load_adult_age5() # replace function if other Dataset wanted
    elif LOADER == 'random':
        data_dict, X_test, y_test, s_list, _, _ = DatasetLoader.load_adult_data.load_adult_random() # replace function if other Dataset wanted
    else:
        raise ValueError(f"Unknown loader: {LOADER}")

    input_dim = X_test.shape[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    #Setup Strategies based on ALGORITHM
    if ALGORITHM == 'FedAvg':
        client_loss = lossStrategies.loss_standard
        server_agg  = lossStrategies.agg_fedavg
        
    else:
        raise ValueError(f"Unknown Algorithm: {ALGORITHM}")

    #Initialize Server and Clients
    server = FLServer((X_test, y_test, s_list), input_dim, device)
    
    clients = []
    for c_name, c_data in data_dict.items():
        clients.append(FLClient(c_name, c_data, input_dim, device))

    maj = (y_val.mean() >= 0.5).float()
    baseline = (y_val == maj).float().mean().item()
    print("Majority baseline acc:", baseline)

    print(f"Initialized {len(clients)} clients.")

    #Simulation Loop
    for r in range(ROUNDS):
        print(f"\n--- Round {r+1} ---")
        
        #Get Global State
        global_weights = server.broadcast_weights()
        
        #Train Clients
        client_reports = []
        
        for client in clients:
            #Load global weights
            client.set_parameters(global_weights)
            
            
            # Train
            report = client.train(
                epochs=CLIENT_EPOCHS, 
                lr=LR, 
                loss_strategy=client_loss, 
            )
            client_reports.append(report)
            
        #Server Aggregation
        server.aggregate(client_reports, server_agg)

        #Evaluate
        metrics = server.evaluate()
        print(f"Results Round {r+1}: Acc: {metrics['Accuracy']:.4f}")

if __name__ == "__main__":
    runFLSimulation()