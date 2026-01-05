import torch
import numpy as np
import DatasetLoader.load_adult_data, DatasetLoader.load_acs_data, DatasetLoader.load_bank_data, DatasetLoader.load_kdd_data, DatasetLoader.load_cac_data
from client import FLClient
from server import FLServer
from logger import FLLogger
import lossStrategies

# --- Configuration ---
ALGORITHM = 'FedAvg'  # Options: 'FedAvg', 'FedMinMax', 'TrustFed', 'Fairness' / only FedAvg currently implemented
LOADER = '3_clients'     # Options: '3_clients', '5_clients', 'random'
ROUNDS = 10
CLIENT_EPOCHS = 5
LR = 0.01

def runFLSimulation():
    print(f"--- Starting FL Simulation: {ALGORITHM} on {LOADER} ---")

   #Initialize Logger
    logger = FLLogger(
        algorithm=ALGORITHM,
        loader=LOADER,
        config={
            "rounds": ROUNDS,
            "client_epochs": CLIENT_EPOCHS,
            "learning_rate": LR
        }
    )
    
    #Load Data
    print("Loading data...")
    if LOADER == '3_clients':
        data_dict, X_test, y_test, s_list, _, _ = DatasetLoader.load_adult_data.load_adult_age3("./Datasets/adult.csv", "sex") # replace function if other Dataset wanted
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
        
    print(f"Initialized {len(clients)} clients.")

    #Track best round per metric
    best_acc_value = -1.0
    best_round_acc = -1

    best_sp_value = float('inf')
    best_round_sp = -1

    best_eo_value = float('inf')
    best_round_eo = -1

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


            #Train
            report = client.train(epochs=CLIENT_EPOCHS, lr=LR, loss_strategy=client_loss)
            client_reports.append(report)

        #Log client data
        logger.log_clients(r + 1, client_reports)

        #Server Aggregation
        server.aggregate(client_reports, server_agg)

        #Evaluate
        metrics = server.evaluate()
        logger.log_round(r + 1, metrics)

        #Track best round for each metric
        if metrics["Accuracy"] > best_acc_value:
            best_acc_value = metrics["Accuracy"]
            best_round_acc = r + 1

        if abs(metrics["Statistical_Parity"]) < best_sp_value:
            best_sp_value = abs(metrics["Statistical_Parity"])
            best_round_sp = r + 1

        if metrics["Equalized_Odds"] < best_eo_value:
            best_eo_value = metrics["Equalized_Odds"]
            best_round_eo = r + 1

        print(
            f"Results Round {r+1}: "
            f"Acc={metrics['Accuracy']:.4f}, "
            f"SP={metrics['Statistical_Parity']:.4f}, "
            f"EO={metrics['Equalized_Odds']:.4f}"
        )
    #Save best metrics to CSV and .log

    logger.best_metrics = {
    "Accuracy": {"round": best_round_acc, "value": best_acc_value},
    "Statistical_Parity": {"round": best_round_sp, "value": best_sp_value},
    "Equalized_Odds": {"round": best_round_eo, "value": best_eo_value}
}
    logger.finalize()

    #Summary
    print("\nBest Rounds Summary:")
    for metric, info in logger.best_metrics.items():
        print(f"{metric}: Round {info['round']} | Value: {info['value']:.4f}")
    


if __name__ == "__main__":
    runFLSimulation()