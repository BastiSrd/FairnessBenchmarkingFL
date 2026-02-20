import torch
import numpy as np
from DatasetLoader import load_adult_data, load_acs_data, load_bank_data, load_kdd_data, load_cac_data
from FedAvg.FedAvgClient import FedAvgClient
from FedAvg.FedAvgServer import FedAvgServer
from logger import FLLogger
from FedAvg.FedAvgLossStrategies import loss_standard, agg_fedavg
from EOFedMinMax.EOFedMinMaxClient import EOFedMinMaxClient
from EOFedMinMax.EOFedMinMaxServer import EOFedMinMaxServer
from EOFedMinMax.lossStrategiesEOFedMinMax import agg_EOfedminmax, loss_EOfedminmax
from OrigFedMinMax.OrigFedMinMaxClient import OrigFedMinMaxClient
from OrigFedMinMax.OrigFedMinMaxServer import OrigFedMinMaxServer
from OrigFedMinMax.OriglossStrategiesFedMinMax import agg_Origfedminmax, loss_Origfedminmax

# --- Configuration ---
ALGORITHM = 'EOFedMinMax'  # Options: 'FedAvg', 'FedMinMax', 'TrustFed', 'Fairness' / only FedAvg currently implemented
LOADER = '5_clients'     # Options: '3_clients', '5_clients', 'random'
ROUNDS = 50
CLIENT_EPOCHS = 1
LR = 0.005


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
        data_dict, X_test, y_test, s_list, cols, ypot, X_val, y_val, sval_list, yvalpot = load_adult_data.load_adult_age3() # replace function if other Dataset wanted
    elif LOADER == '5_clients':
        data_dict, X_test, y_test, s_list, cols, ypot, X_val, y_val, sval_list, yvalpot  = load_bank_data.load_bank_age_5() # replace function if other Dataset wanted
    elif LOADER == 'random':
        data_dict, X_test, y_test, s_list, cols, ypot, X_val, y_val, sval_list, yvalpot  = load_adult_data.load_adult_random(10) # replace function if other Dataset wanted
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
    elif ALGORITHM == "EOFedMinMax":
        client_loss = loss_EOfedminmax
        server_agg = agg_EOfedminmax
        server = EOFedMinMaxServer((X_test, y_test, s_list, X_val, y_val, sval_list), input_dim, device)
        clients = []
        for c_name, c_data in data_dict.items():
            clients.append(EOFedMinMaxClient(c_name, c_data, input_dim, device))
        print(f"Initialized {len(clients)} clients.")

        runEOFedMinMaxSimulationLoop(server,clients,logger,client_loss,server_agg, data_dict)
    elif ALGORITHM == "FedMinMax":
        client_loss = loss_Origfedminmax
        server_agg = agg_Origfedminmax
        s_test_tensor = torch.tensor(s_list).long()
        y_test_tensor = y_test.view(-1).long()
        s_joint_test = (s_test_tensor * 2 + y_test_tensor).tolist()
        s_val_tensor = torch.tensor(sval_list).long()
        y_val_tensor = y_val.view(-1).long()
        s_joint_val = (s_val_tensor * 2 + y_val_tensor).tolist() 
        server = OrigFedMinMaxServer((X_test, y_test, s_joint_test, X_val, y_val, s_joint_val), input_dim, device)
        clients = []
        for c_name, c_data in data_dict.items():
            clients.append(OrigFedMinMaxClient(c_name, c_data, input_dim, device))
        print(f"Initialized {len(clients)} clients.")

        runOrigFedMinMaxSimulationLoop(server,clients,logger,client_loss,server_agg, data_dict)
    else:
        raise ValueError(f"Unknown Algorithm: {ALGORITHM}")

def runOrigFedMinMaxSimulationLoop(server, clients, logger, client_loss, server_agg, data_dict):
    #Track best round per metric

    best_blAcc_value = -1.0
    best_round_blAcc = -1

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


        #Evaluate
        metrics = server.evaluate()
        logger.log_round(r + 1, metrics)

        #Track best round for each metric
        if metrics["Accuracy"] > best_acc_value:
            best_acc_value = metrics["Accuracy"]
            best_round_acc = r + 1

        if metrics["balanced_Accuracy"] > best_blAcc_value:
            best_blAcc_value = metrics["balanced_Accuracy"]
            best_round_blAcc = r + 1

        if abs(metrics["Statistical_Parity"]) < best_sp_value and abs(metrics["Statistical_Parity"]) > 0:
            best_sp_value = abs(metrics["Statistical_Parity"])
            best_round_sp = r + 1

        if metrics["Equalized_Odds"] < best_eo_value and metrics["Equalized_Odds"] > 0:
            best_eo_value = metrics["Equalized_Odds"]
            best_round_eo = r + 1

        print(
            f"Results Round {r+1}: "
            f"Acc={metrics['Accuracy']:.4f}, "
            f"Balanced Acc={metrics['balanced_Accuracy']:.4f}, "
            f"SP={metrics['Statistical_Parity']:.4f}, "
            f"EO={metrics['Equalized_Odds']:.4f}"
        )
    
    #Apply the averaged model as the final test
    server.load_averaged_model()
    metrics = server.evaluate()
    logger.log_round(r+2, metrics)

    #Track best round for each metric
    if metrics["Accuracy"] > best_acc_value:
        best_acc_value = metrics["Accuracy"]
        best_round_acc = r + 2

    if metrics["balanced_Accuracy"] > best_blAcc_value:
        best_blAcc_value = metrics["balanced_Accuracy"]
        best_round_blAcc = r + 2

    if abs(metrics["Statistical_Parity"]) < best_sp_value:
        best_sp_value = abs(metrics["Statistical_Parity"])
        best_round_sp = r+2

    if metrics["Equalized_Odds"] < best_eo_value:
        best_eo_value = metrics["Equalized_Odds"]
        best_round_eo = r+2

    print(
            f"Results Round {r+2}: "
            f"Acc={metrics['Accuracy']:.4f}, "
            f"Balanced Acc={metrics['balanced_Accuracy']:.4f}, "
            f"SP={metrics['Statistical_Parity']:.4f}, "
            f"EO={metrics['Equalized_Odds']:.4f}"
        )


    #Save best metrics to CSV and .log
    logger.best_metrics = {
    "Accuracy": {"round": best_round_acc, "value": best_acc_value},
    "balanced_Accuracy": {"round": best_round_blAcc, "value": best_blAcc_value},
    "Statistical_Parity": {"round": best_round_sp, "value": best_sp_value},
    "Equalized_Odds": {"round": best_round_eo, "value": best_eo_value}
    }
    logger.finalize()

    #Summary
    print("\nBest Rounds Summary:")
    for metric, info in logger.best_metrics.items():
        print(f"{metric}: Round {info['round']} | Value: {info['value']:.10f}")



def runEOFedMinMaxSimulationLoop(server, clients, logger, client_loss, server_agg, data_dict):
    #Track best round per metric
    best_blAcc_value = -1.0
    best_round_blAcc = -1

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

        uniq = torch.unique(s)
        for gid in uniq.tolist():
            gid = int(gid)
            mask_g = (s == gid)
            n0 = int(((y == 0) & mask_g).sum().item())
            n1 = int(((y == 1) & mask_g).sum().item())

            if gid not in global_group_counts:
                global_group_counts[gid] = {"y0": 0, "y1": 0}
            global_group_counts[gid]["y0"] += n0
            global_group_counts[gid]["y1"] += n1

    print(f"global_group_counts (EO): {global_group_counts}")
    server.set_global_stats(global_group_counts)

    
    for r in range(ROUNDS):
        print(f"\n--- Round {r+1} ---")

        #Get Global State
        broadcast_data = server.initializeWeights()
        global_weights = broadcast_data['model_weights']
        w_y0 = broadcast_data.get('group_weights_y0', {})
        w_y1 = broadcast_data.get('group_weights_y1', {})
        lambda_eo = broadcast_data.get('lambda_eo', 0.0)


        #Train Clients
        client_reports = []
        for client in clients:
            #Load global weights
            client.set_parameters(global_weights)


            #Train
            report = client.train(
                epochs=CLIENT_EPOCHS,
                lr=LR,
                loss_strategy=client_loss,
                strategy_context={
                    'group_weights_y0': w_y0,
                    'group_weights_y1': w_y1,
                    'lambda_eo': lambda_eo
                }
            )
            client_reports.append(report)

        #Log client data
        logger.log_clients(r + 1, client_reports)

        #Server Aggregation
        server.aggregate(client_reports, server_agg)


        #Evaluate
        metrics = server.evaluate(final=False)
        logger.log_round(r + 1, metrics)

        #Track best round for each metric
        if metrics["Accuracy"] > best_acc_value:
            best_acc_value = metrics["Accuracy"]
            best_round_acc = r + 1

        if metrics["balanced_Accuracy"] > best_blAcc_value:
            best_blAcc_value = metrics["balanced_Accuracy"]
            best_round_blAcc = r + 1

        if abs(metrics["Statistical_Parity"]) < best_sp_value:
            best_sp_value = abs(metrics["Statistical_Parity"])
            best_round_sp = r + 1

        if metrics["Equalized_Odds"] < best_eo_value:
            best_eo_value = metrics["Equalized_Odds"]
            best_round_eo = r + 1

        print(
            f"Results Round {r+1}: "
            f"Acc={metrics['Accuracy']:.8f}, "
            f"Balanced Acc={metrics['balanced_Accuracy']:.8f}, "
            f"SP={metrics['Statistical_Parity']:.8f}, "
            f"EO={metrics['Equalized_Odds']:.8f}"
        )
    
    #Apply the averaged model as the final test
    server.load_averaged_model()
    metrics = server.evaluate(final= True)
    logger.log_round(r+2, metrics)

    #Track best round for each metric
    if metrics["Accuracy"] > best_acc_value:
        best_acc_value = metrics["Accuracy"]
        best_round_acc = r+2

    if metrics["balanced_Accuracy"] > best_blAcc_value:
        best_blAcc_value = metrics["balanced_Accuracy"]
        best_round_blAcc = r + 2

    if abs(metrics["Statistical_Parity"]) < best_sp_value:
        best_sp_value = abs(metrics["Statistical_Parity"])
        best_round_sp = r+2

    if metrics["Equalized_Odds"] < best_eo_value:
        best_eo_value = metrics["Equalized_Odds"]
        best_round_eo = r+2

    print(
            f"Results Round {r+2}: "
            f"Acc={metrics['Accuracy']:.8f}, "
            f"Balanced Acc={metrics['balanced_Accuracy']:.8f}, "
            f"SP={metrics['Statistical_Parity']:.8f}, "
            f"EO={metrics['Equalized_Odds']:.8f}"
        )


    #Save best metrics to CSV and .log
    logger.best_metrics = {
    "Accuracy": {"round": best_round_acc, "value": best_acc_value},
    "balanced_Accuracy": {"round": best_round_blAcc, "value": best_blAcc_value},
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

    best_blAcc_value = -1.0
    best_round_blAcc = -1

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
        
        if metrics["balanced_Accuracy"] > best_blAcc_value:
            best_blAcc_value = metrics["balanced_Accuracy"]
            best_round_blAcc = r + 1

        if abs(metrics["Statistical_Parity"]) < best_sp_value:
            best_sp_value = abs(metrics["Statistical_Parity"])
            best_round_sp = r + 1

        if metrics["Equalized_Odds"] < best_eo_value:
            best_eo_value = metrics["Equalized_Odds"]
            best_round_eo = r + 1

        print(
            f"Results Round {r+1}: "
            f"Acc={metrics['Accuracy']:.4f}, "
            f"Balanced Acc={metrics['balanced_Accuracy']:.4f}, "
            f"SP={metrics['Statistical_Parity']:.4f}, "
            f"EO={metrics['Equalized_Odds']:.4f}"
        )
    #Save best metrics to CSV and .log

    logger.best_metrics = {
    "Accuracy": {"round": best_round_acc, "value": best_acc_value},
    "balanced_Accuracy": {"round": best_round_blAcc, "value": best_blAcc_value},
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