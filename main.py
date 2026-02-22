from typing import Optional
import torch

import DatasetLoader.load_acs_data
import DatasetLoader.load_adult_data
import DatasetLoader.load_bank_data
import DatasetLoader.load_cac_data
import DatasetLoader.load_kdd_data
from FedAvg.FedAvgClient import FedAvgClient
from FedAvg.FedAvgLossStrategies import loss_standard, agg_fedavg
from FedAvg.FedAvgServer import FedAvgServer
from TrustFed.TrustFedClient import TrustFedClient
from TrustFed.TrustFedLoss import loss_trustfed, agg_trustfed
from TrustFed.TrustFedServer import TrustFedServer
from EOFedMinMax.EOFedMinMaxClient import EOFedMinMaxClient
from EOFedMinMax.lossStrategiesEOFedMinMax import loss_EOfedminmax, agg_EOfedminmax
from EOFedMinMax.EOFedMinMaxServer import EOFedMinMaxServer
from OrigFedMinMax.OrigFedMinMaxClient import OrigFedMinMaxClient
from OrigFedMinMax.OriglossStrategiesFedMinMax import loss_Origfedminmax, agg_Origfedminmax
from OrigFedMinMax.OrigFedMinMaxServer import OrigFedMinMaxServer
from logger import FLLogger

# --- Configuration ---
#ALGORITHM = 'TrustFed'  # Options: 'FedAvg', 'FedMinMax', "EOFedMinMax" ,'TrustFed'
#LOADER = '3_clients'  # Options: '3_clients', '5_clients', 'random'

# --- Trainingskonstanten for Benchmark
ROUNDS = 50
CLIENT_EPOCHS = 1
LR = 0.005


# defaults mentioned in paper
TRUSTFED_ALPHA = 1.0
TRUSTFED_P_NORM = 2


def runFLSimulation(loaderID, algorithmID, trustfedFairness: Optional[str] = None):
    loader_map = {
    # Adult
        "adult_iid5": lambda: DatasetLoader.load_adult_data.load_adult_random(num_clients=5),
        "adult_iid10": lambda: DatasetLoader.load_adult_data.load_adult_random(num_clients=10),
        "adult_age3": DatasetLoader.load_adult_data.load_adult_age3,
        "adult_age5": DatasetLoader.load_adult_data.load_adult_age5,
        # Bank
        "bank_iid5": lambda: DatasetLoader.load_bank_data.load_bank_random(num_clients=5),
        "bank_iid10": lambda: DatasetLoader.load_bank_data.load_bank_random(num_clients=10),
        "bank_age3": DatasetLoader.load_bank_data.load_bank_age3,
        "bank_age5": DatasetLoader.load_bank_data.load_bank_age_5, 
        # KDD
        "kdd_iid5": lambda: DatasetLoader.load_kdd_data.load_kdd_random(num_clients=5),
        "kdd_iid10": lambda: DatasetLoader.load_kdd_data.load_kdd_random(num_clients=10),
        "kdd_age3": DatasetLoader.load_kdd_data.load_kdd_age3,
        "kdd_age5": DatasetLoader.load_kdd_data.load_kdd_age5,
        # ACS
        "acs_iid5": lambda: DatasetLoader.load_acs_data.load_acs_random(num_clients=5),
        "acs_iid10": lambda: DatasetLoader.load_acs_data.load_acs_random(num_clients=10),
        "acs_state3": DatasetLoader.load_acs_data.load_acs_states_3,
        "acs_state5": DatasetLoader.load_acs_data.load_acs_states_5,
        # CAC
        "cac_iid5": lambda: DatasetLoader.load_cac_data.load_cac_random(num_clients=5),
        "cac_iid10": lambda: DatasetLoader.load_cac_data.load_cac_random(num_clients=10),
        "cac_state3": DatasetLoader.load_cac_data.load_cac_states_3,
        "cac_state5": DatasetLoader.load_cac_data.load_cac_states_5,
    }


    print(f"--- Starting FL Simulation: {algorithmID} on {loaderID} ---")

   #Initialize Logger
    if algorithmID == "TrustFed":
        logger = FLLogger(
            algorithm=algorithmID+"_"+trustfedFairness,
            loader=loaderID,
            config={
                "rounds": ROUNDS,
                "client_epochs": CLIENT_EPOCHS,
                "learning_rate": LR,
            }
        )
    else:
        logger = FLLogger(
            algorithm=algorithmID,
            loader=loaderID,
            config={
                "rounds": ROUNDS,
                "client_epochs": CLIENT_EPOCHS,
                "learning_rate": LR
            }
        )
    
    data_dict, X_test, y_test, s_list, cols, ypot, X_val, y_val, sval_list, yvalpot = loader_map[loaderID]()
    input_dim = X_test.shape[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")


    #Setup Strategies and FL Environment based on ALGORITHM
    if algorithmID == 'FedAvg':
        client_loss = loss_standard
        server_agg  = agg_fedavg
        server = FedAvgServer((X_test, y_test, s_list,X_val, y_val, sval_list), input_dim, device)
        clients = []
        for c_name, c_data in data_dict.items():
            clients.append(FedAvgClient(c_name, c_data, input_dim, device))
        print(f"Initialized {len(clients)} clients.")

        runFedAvgSimulationLoop(server,clients,logger,client_loss,server_agg)
    elif algorithmID == "EOFedMinMax":
        client_loss = loss_EOfedminmax
        server_agg = agg_EOfedminmax
        server = EOFedMinMaxServer((X_test, y_test, s_list, X_val, y_val, sval_list), input_dim, device)
        clients = []
        for c_name, c_data in data_dict.items():
            clients.append(EOFedMinMaxClient(c_name, c_data, input_dim, device))
        print(f"Initialized {len(clients)} clients.")

        runEOFedMinMaxSimulationLoop(server,clients,logger,client_loss,server_agg, data_dict)
    elif algorithmID == "FedMinMax":
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


    elif algorithmID == "TrustFed":
        _objectives, _metrics = run_trustfed_once(
            alpha=TRUSTFED_ALPHA,
            lr=LR,
            rounds=ROUNDS,
            client_epochs=CLIENT_EPOCHS,
            data_dict=data_dict,
            X_test=X_test,
            y_test=y_test,
            s_list=s_list,
            input_dim=input_dim,
            device=device,
            logger=logger,
            fairness_notion=trustfedFairness,
            p_norm=TRUSTFED_P_NORM,
            x_val=X_val,
            y_val=y_val,
            s_val=sval_list
        )

    else:
        raise ValueError(f"Unknown Algorithm: {algorithmID}")

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
        metrics = server.evaluate(final=False)
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
    metrics = server.evaluate(final=True)
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

    best_balanced_acc_value = -1.0
    best_round_balanced_acc = -1

    best_acc_value = -1.0
    best_round_acc = -1

    best_sp_value = float('inf')
    best_round_sp = -1

    best_eo_value = float('inf')
    best_round_eo = -1

    best_sp_value = float('inf')
    best_round_sp = -1

    best_eo_value = float('inf')
    best_round_eo = -1

    bestWeights = server.broadcast_weights()

    # Simulation Loop
    for r in range(ROUNDS):
        print(f"\n--- Round {r + 1} ---")

        # Get Global State
        global_weights = server.broadcast_weights()

        # Train Clients
        client_reports = []
        for client in clients:
            # Load global weights
            client.set_parameters(global_weights)

            # Train
            report = client.train(epochs=CLIENT_EPOCHS, lr=LR, loss_strategy=client_loss)
            client_reports.append(report)

        # Log client data
        logger.log_clients(r + 1, client_reports)

        # Server Aggregation
        server.aggregate(client_reports, server_agg)

        # Evaluate
        metrics = server.evaluate(final=False)

        logger.log_round(r + 1, metrics)

        # Track best round for each metric
        if metrics["balanced_Accuracy"] > best_balanced_acc_value:
            best_balanced_acc_value = metrics["balanced_Accuracy"]
            best_round_balanced_acc = r + 1

        if metrics["Accuracy"] > best_acc_value:
            best_acc_value = metrics["Accuracy"]
            best_round_acc = r + 1
            bestWeights = server.broadcast_weights()

        if abs(metrics["Statistical_Parity"]) < best_sp_value:
            best_sp_value = abs(metrics["Statistical_Parity"])
            best_round_sp = r + 1

        if metrics["Equalized_Odds"] < best_eo_value:
            best_eo_value = metrics["Equalized_Odds"]
            best_round_eo = r + 1

        print(
            f"Results Round {r + 1}: "
            f"Balanced_Acc={metrics['balanced_Accuracy']:.4f}, "
            f"Acc={metrics['Accuracy']:.4f}, "
            f"SP={metrics['Statistical_Parity']:.4f}, "
            f"EO={metrics['Equalized_Odds']:.4f}"
        )
    # Save best metrics to CSV and .log
    server.loadBestModel(bestWeights)
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

    logger.best_metrics = {
        "Balanced_Accuracy": {"round": best_round_balanced_acc, "value": best_balanced_acc_value},
        "Accuracy": {"round": best_round_acc, "value": best_acc_value},
        "Statistical_Parity": {"round": best_round_sp, "value": best_sp_value},
        "Equalized_Odds": {"round": best_round_eo, "value": best_eo_value}
    }
    logger.finalize()

    # Summary
    print("\nBest Rounds Summary:")
    for metric, info in logger.best_metrics.items():
        print(f"{metric}: Round {info['round']} | Value: {info['value']:.4f}")


def run_trustfed_once(
        alpha: float,
        lr: float,
        rounds: int,
        client_epochs: int,
        data_dict,
        X_test,
        y_test,
        s_list,
        x_val,
        y_val, 
        s_val,
        input_dim: int,
        device: str,
        logger: FLLogger,
        fairness_notion: str = "SP",
        p_norm: int = 2,
):
    """
    Run a single TrustFed simulation (multi-round federated training + evaluation).

    Parameters:
        alpha (float): Fairness constraint weight passed into the TrustFed constraint loss.
        lr (float): Client optimizer learning rate.
        rounds (int): Number of federated communication rounds.
        client_epochs (int): Number of local epochs per client per round.
        data_dict (dict): Mapping client_name -> client data dict with tensors:
            - "X": features (N, D)
            - "y": labels (N,)
            - "s": sensitive attributes (N,)
        X_test (torch.Tensor): Global test features, shape (N_test, D).
        y_test (torch.Tensor): Global test labels, shape (N_test,).
        s_list (torch.Tensor or list): Global test sensitive attributes, shape (N_test,).
        input_dim (int): Input feature dimension D.
        device (str): Device used for training/evaluation ("cpu" or "cuda").
        logger (FLLogger): Logger used to record client reports and round metrics.
        fairness_notion (str): Fairness notion for constraints ("SP" or "EO").
        p_norm (int): Norm used inside the TrustFed constraint loss (typically 2).

    Returns:
        tuple: (None, last_metrics) where last_metrics is the evaluation dict from the final round.
    """

    server = TrustFedServer(
        (X_test, y_test, s_list, x_val, y_val, s_val),
        input_dim,
        device,
        fairness_notion=fairness_notion,
        alpha=alpha,
    )

    clients = [TrustFedClient(c_name, c_data, input_dim, device) for c_name, c_data in data_dict.items()]
    print(f"Initialized {len(clients)} clients.")

    # Track best rounds
    best_balanced_acc_value = -1.0
    best_round_balanced_acc = -1

    best_acc_value = -1.0
    best_round_acc = -1

    best_sp_value = float('inf')
    best_round_sp = -1

    best_eo_value = float('inf')
    best_round_eo = -1

    last_metrics = None

    bestWeights = server.broadcast_weights()

    for r in range(rounds):
        print(f"\n--- Round {r + 1} ---")
        global_weights = server.broadcast_weights()

        client_reports = []
        for client in clients:
            client.set_parameters(global_weights)
            report = client.train(
                epochs=client_epochs,
                lr=lr,
                loss_strategy=loss_trustfed,
                strategy_context={
                    "fairness_notion": fairness_notion,
                    "alpha": alpha,
                    "p_norm": p_norm,
                    "sensitive_classes": [0, 1],
                }
            )
            client_reports.append(report)

        logger.log_clients(r + 1, client_reports)
        server.aggregate(client_reports, agg_trustfed)

        metrics = server.evaluate(final=False)
        last_metrics = metrics
        logger.log_round(r + 1, metrics)

        if metrics["balanced_Accuracy"] > best_balanced_acc_value:
            best_balanced_acc_value = metrics["balanced_Accuracy"]
            best_round_balanced_acc = r + 1

        if metrics["Accuracy"] > best_acc_value:
            best_acc_value = metrics["Accuracy"]
            best_round_acc = r + 1
            bestWeights = server.broadcast_weights()

        if abs(metrics["Statistical_Parity"]) < best_sp_value:
            best_sp_value = abs(metrics["Statistical_Parity"])
            best_round_sp = r + 1

        if metrics["Equalized_Odds"] < best_eo_value:
            best_eo_value = metrics["Equalized_Odds"]
            best_round_eo = r + 1

        print(
            f"Results Round {r + 1}: "
            f"Balanced_Acc={metrics['balanced_Accuracy']:.4f}, "
            f"Acc={metrics['Accuracy']:.4f}, "
            f"SP={metrics['Statistical_Parity']:.10f}, "
            f"EO={metrics['Equalized_Odds']:.10f}"
        )

    server.loadBestModel(bestWeights)
    metrics = server.evaluate(final= True)
    logger.log_round(r+2, metrics)

    if metrics["balanced_Accuracy"] > best_balanced_acc_value:
            best_balanced_acc_value = metrics["balanced_Accuracy"]
            best_round_balanced_acc = r + 2

    if metrics["Accuracy"] > best_acc_value:
        best_acc_value = metrics["Accuracy"]
        best_round_acc = r + 2
        bestWeights = server.broadcast_weights()

    if abs(metrics["Statistical_Parity"]) < best_sp_value:
        best_sp_value = abs(metrics["Statistical_Parity"])
        best_round_sp = r + 2

    if metrics["Equalized_Odds"] < best_eo_value:
        best_eo_value = metrics["Equalized_Odds"]
        best_round_eo = r + 2

    print(
        f"Results Round {r + 2}: "
        f"Balanced_Acc={metrics['balanced_Accuracy']:.4f}, "
        f"Acc={metrics['Accuracy']:.4f}, "
        f"SP={metrics['Statistical_Parity']:.10f}, "
        f"EO={metrics['Equalized_Odds']:.10f}"
    )

    logger.best_metrics = {
        "Balanced_Accuracy": {"round": best_round_balanced_acc, "value": best_balanced_acc_value},
        "Accuracy": {"round": best_round_acc, "value": best_acc_value},
        "Statistical_Parity": {"round": best_round_sp, "value": best_sp_value},
        "Equalized_Odds": {"round": best_round_eo, "value": best_eo_value}
    }
    logger.finalize()

    return None, last_metrics


if __name__ == "__main__":

    
    LOADERS = [
    "adult_iid5", "adult_iid10", "adult_age3", "adult_age5", 
    "bank_iid5", "bank_iid10", "bank_age3", "bank_age5", 
    "kdd_iid5", "kdd_iid10", "kdd_age3", "kdd_age5", 
    "acs_iid5", "acs_iid10", "acs_state3", "acs_state5", 
    "cac_iid5", "cac_iid10", "cac_state3", "cac_state5"
    ]
    ALGORITHMS = ['FedAvg', 'FedMinMax', "EOFedMinMax", 'TrustFed']
    TRUSTFED_FAIRNESS = ["SP", "EO"]  # "SP" or "EO"
    NUM_RUNS = 3
    for run_idx in range(NUM_RUNS):
        print(f"=== Starting Experimental Run {run_idx + 1}/{NUM_RUNS} ===")
        for algorithm in ALGORITHMS:
            if algorithm == "TrustFed":
                for loader in LOADERS:
                    for fairness in TRUSTFED_FAIRNESS:
                        runFLSimulation(loaderID=loader, algorithmID=algorithm, trustfedFairness=fairness)
            else:
                for loader in LOADERS:
                    runFLSimulation(loaderID=loader, algorithmID=algorithm)

