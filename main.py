import torch
import numpy as np
import DatasetLoader.load_adult_data, DatasetLoader.load_acs_data, DatasetLoader.load_bank_data, DatasetLoader.load_kdd_data, DatasetLoader.load_cac_data
from FedAvg.FedAvgClient import FedAvgClient
from FedAvg.FedAvgServer import FedAvgServer
from logger import FLLogger
from FedAvg.FedAvgLossStrategies import loss_standard, agg_fedavg
from FedMinMax.FedMinMaxClient import FedMinMaxClient
from FedMinMax.FedMinMaxServer import FedMinMaxServer
from FedMinMax.lossStrategiesFedMinMax import agg_fedminmax, loss_fedminmax
from EOFedMinMax.EOFedMinMaxClient import EOFedMinMaxClient
from EOFedMinMax.EOFedMinMaxServer import EOFedMinMaxServer
from EOFedMinMax.EOlossStrategiesFedMinMax import agg_EOfedminmax, loss_EOfedminmax

# --- Configuration ---
ALGORITHM = 'FedMinMax'  # Options: 'FedAvg', 'FedMinMax', 'TrustFed', 'Fairness' / only FedAvg currently implemented
LOADER = '3_clients'     # Options: '3_clients', '5_clients', 'random'
ROUNDS = 25
CLIENT_EPOCHS = 1
LR = 0.001


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


    #Setup Strategies and FL Environment based on ALGORITHM
    if ALGORITHM == 'FedAvg':
        client_loss = loss_standard
        server_agg  = agg_fedavg
        server = FedAvgServer((X_test, y_test, s_list), input_dim, device)
        clients = []
        for c_name, c_data in data_dict.items():
            clients.append(FedAvgClient(c_name, c_data, input_dim, device))
        print(f"Initialized {len(clients)} clients.")

        runFedAvgSimulationLoop(server,clients,logger,client_loss,server_agg)
    elif ALGORITHM == "FedMinMax":
        client_loss = loss_fedminmax
        server_agg = agg_fedminmax
        server = FedMinMaxServer((X_test, y_test, s_list), input_dim, device)
        clients = []
        for c_name, c_data in data_dict.items():
            clients.append(FedMinMaxClient(c_name, c_data, input_dim, device))
        print(f"Initialized {len(clients)} clients.")

        runFedMinMaxSimulationLoop(server,clients,logger,client_loss,server_agg, data_dict)
    elif ALGORITHM == "EOFedMinMax":
        client_loss = loss_EOfedminmax
        server_agg = agg_EOfedminmax
        s_test_tensor = torch.tensor(s_list).long()
        y_test_tensor = y_test.view(-1).long()
        s_joint_test = (s_test_tensor * 2 + y_test_tensor).tolist()     
        server = EOFedMinMaxServer((X_test, y_test, s_joint_test), input_dim, device)
        clients = []
        for c_name, c_data in data_dict.items():
            clients.append(EOFedMinMaxClient(c_name, c_data, input_dim, device))
        print(f"Initialized {len(clients)} clients.")

        runEOFedMinMaxSimulationLoop(server,clients,logger,client_loss,server_agg, data_dict)
    else:
        raise ValueError(f"Unknown Algorithm: {ALGORITHM}")

def runEOFedMinMaxSimulationLoop(server, clients, logger, client_loss, server_agg, data_dict):
    #Track best round per metric
    best_acc_value = -1.0
    best_round_acc = -1

    best_sp_value = float('inf')
    best_round_sp = -1

    best_eo_value = float('inf')
    best_round_eo = -1
    
    
    # [FEDMINMAX SETUP]
    # Calculate global group counts to set priors (rho) in Server
    global_group_counts = {}
    for c_data in data_dict.values():
        s = c_data['s'].view(-1).long()
        y = c_data['y'].view(-1).long()
        
        # Map (A,Y) to {0, 1, 2, 3}
        s_joint = s * 2 + y
        
        uniques, counts = torch.unique(s_joint, return_counts=True)
        for gid, c in zip(uniques.tolist(), counts.tolist()):
            global_group_counts[gid] = global_group_counts.get(gid, 0) + c
    print(f"global_group_counts: {global_group_counts}")

    # FedMinMax - Pass stats to server for correct rho calculation
    server.set_global_stats(global_group_counts)

    
    for r in range(ROUNDS):
        print(f"\n--- Round {r+1} ---")

        #Get Global State
        broadcast_data = server.initializeWeights()
        global_weights = broadcast_data['model_weights']
        group_weights = broadcast_data.get('group_weights', {})


        #Train Clients
        client_reports = []
        for client in clients:
            #Load global weights
            client.set_parameters(global_weights)


            #Train
            report = client.train(epochs=CLIENT_EPOCHS, lr=LR, loss_strategy=client_loss, strategy_context={'group_weights': group_weights})
            client_reports.append(report)

        #Log client data
        logger.log_clients(r + 1, client_reports)

        #Server Aggregation
        server.aggregate(client_reports, server_agg)

        # After server.aggregate(...)
        risk_vec = None
        if hasattr(server, "mu") and hasattr(server, "group_ids"):
            # If your aggregate passes back debug info some other way, fetch it here.
            # If you stored it in server_state only, easiest is to also store it in server during aggregate.
            pass
        server.log_fedminmax_state(round_idx=r+1, risk_vector=server._last_risk_vector, top_k=10)

        #Evaluate
        metrics = server.evaluate()
        logger.log_round(r + 1, metrics)

        #Track best round for each metric
        if metrics["Accuracy"] > best_acc_value:
            best_acc_value = metrics["Accuracy"]
            best_round_acc = r + 1

        if abs(metrics["Statistical_Parity"]) < best_sp_value and abs(metrics["Statistical_Parity"]) > 0:
            best_sp_value = abs(metrics["Statistical_Parity"])
            best_round_sp = r + 1

        if metrics["Equalized_Odds"] < best_eo_value and metrics["Equalized_Odds"] > 0:
            best_eo_value = metrics["Equalized_Odds"]
            best_round_eo = r + 1

        print(
            f"Results Round {r+1}: "
            f"Acc={metrics['Accuracy']:.4f}, "
            f"SP={metrics['Statistical_Parity']:.10f}, "
            f"EO={metrics['Equalized_Odds']:.10f}"
        )
    
    #Apply the averaged model as the final test
    server.load_averaged_model()
    metrics = server.evaluate()
    logger.log_round(ROUNDS+1, metrics)

    #Track best round for each metric
    if metrics["Accuracy"] > best_acc_value:
        best_acc_value = metrics["Accuracy"]
        best_round_acc = ROUNDS+1

    if abs(metrics["Statistical_Parity"]) < best_sp_value:
        best_sp_value = abs(metrics["Statistical_Parity"])
        best_round_sp = ROUNDS+1

    if metrics["Equalized_Odds"] < best_eo_value:
        best_eo_value = metrics["Equalized_Odds"]
        best_round_eo = ROUNDS+1

    print(
        f"Results Averaged Model: "
        f"Acc={metrics['Accuracy']:.4f}, "
        f"SP={metrics['Statistical_Parity']:.10f}, "
        f"EO={metrics['Equalized_Odds']:.10f}"
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
        print(f"{metric}: Round {info['round']} | Value: {info['value']:.10f}")



def runFedMinMaxSimulationLoop(server, clients, logger, client_loss, server_agg, data_dict):
    #Track best round per metric
    best_acc_value = -1.0
    best_round_acc = -1

    best_sp_value = float('inf')
    best_round_sp = -1

    best_eo_value = float('inf')
    best_round_eo = -1
    
    
    # [FEDMINMAX SETUP]
    # Calculate global group counts to set priors (rho) in Server
    global_group_counts = {}
    for c_data in data_dict.values():
        s = c_data['s'].view(-1).long()
        uniques, counts = torch.unique(s, return_counts=True)
        for u, c in zip(uniques, counts):
            gid = u.item()
            global_group_counts[gid] = global_group_counts.get(gid, 0) + c.item()

    print(f"global group counts: {global_group_counts}")

    # FedMinMax - Pass stats to server for correct rho calculation
    server.set_global_stats(global_group_counts)

    
    for r in range(ROUNDS):
        print(f"\n--- Round {r+1} ---")

        #Get Global State
        broadcast_data = server.initializeWeights()
        global_weights = broadcast_data['model_weights']
        group_weights = broadcast_data.get('group_weights', {})


        #Train Clients
        client_reports = []
        for client in clients:
            #Load global weights
            client.set_parameters(global_weights)


            #Train
            report = client.train(epochs=CLIENT_EPOCHS, lr=LR, loss_strategy=client_loss, strategy_context={'group_weights': group_weights})
            client_reports.append(report)

        #Log client data
        logger.log_clients(r + 1, client_reports)

        #Server Aggregation
        server.aggregate(client_reports, server_agg)

        # After server.aggregate(...)
        risk_vec = None
        if hasattr(server, "mu") and hasattr(server, "group_ids"):
            # If your aggregate passes back debug info some other way, fetch it here.
            # If you stored it in server_state only, easiest is to also store it in server during aggregate.
            pass
        server.log_fedminmax_state(round_idx=r+1, risk_vector=server._last_risk_vector, top_k=10)

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
            f"SP={metrics['Statistical_Parity']:.10f}, "
            f"EO={metrics['Equalized_Odds']:.10f}"
        )
    
    #Apply the averaged model as the final test
    server.load_averaged_model()
    metrics = server.evaluate()
    logger.log_round(ROUNDS+1, metrics)

    #Track best round for each metric
    if metrics["Accuracy"] > best_acc_value:
        best_acc_value = metrics["Accuracy"]
        best_round_acc = ROUNDS+1

    if abs(metrics["Statistical_Parity"]) < best_sp_value:
        best_sp_value = abs(metrics["Statistical_Parity"])
        best_round_sp = ROUNDS+1

    if metrics["Equalized_Odds"] < best_eo_value:
        best_eo_value = metrics["Equalized_Odds"]
        best_round_eo = ROUNDS+1

    print(
        f"Results Averaged Model: "
        f"Acc={metrics['Accuracy']:.4f}, "
        f"SP={metrics['Statistical_Parity']:.10f}, "
        f"EO={metrics['Equalized_Odds']:.10f}"
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
        print(f"{metric}: Round {info['round']} | Value: {info['value']:.10f}")


def runFedAvgSimulationLoop(server, clients, logger,client_loss, server_agg):
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