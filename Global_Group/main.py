# main.py
import os
os.environ["WANDB_MODE"] = "offline"
import random
import torch
import os
import numpy as np
from DatasetLoader.load_adult_data import load_adult_random, load_adult_age3, load_adult_age5
from DatasetLoader.load_bank_data import load_bank_random, load_bank_age3, load_bank_age_5
from DatasetLoader.load_acs_data import load_acs_random, load_acs_states_3, load_acs_states_5
from DatasetLoader.load_kdd_data import load_kdd_random, load_kdd_age3, load_kdd_age5
from DatasetLoader.load_cac_data import load_cac_random,load_cac_states_3, load_cac_states_5
from Global_Group.server import Server
from Global_Group.clients import Client
from Global_Group.yset import YSet
from sklearn.metrics import accuracy_score
from fairnessMetrics import compute_statistical_parity, compute_equalized_odds, compute_balanced_accuracy
from torch.nn import BCELoss
from logger import FLLogger
import argparse
import copy


# -----------------------------
# 1. Model Definition
# -----------------------------
def build_model_class(input_size):
    class NNModel(torch.nn.Module):
        def __init__(self, input_size=input_size, output_size=1):
            super(NNModel, self).__init__()
            self.linear1 = torch.nn.Linear(input_size, 64)
            self.linear2 = torch.nn.Linear(64, output_size)

        def forward(self, x):
            x = torch.relu(self.linear1(x))
            return torch.sigmoid(self.linear2(x)) 
    return NNModel

# -----------------------------
# 2. Hyperparameters
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIENT_EPOCHS = 1
CLIENT_BATCHSIZE = 32
CLIENT_STEPSIZE = 0.005
COMMUNICATION_ROUNDS = 50
NY = 100  # number of points in Y sets for fairness
LAMBDA = 0.2

# -----------------------------
# 3. Dynamic Dataset Selection
# -----------------------------
parser = argparse.ArgumentParser(description="Run Fairness Benchmarking")
parser.add_argument("--loader", type=str, default="adult_iid5", 
                    choices=[
                        "adult_iid5", "adult_iid10", "adult_age3", "adult_age5",
                        "bank_iid5", "bank_iid10", "bank_age3", "bank_age5",
                        "kdd_iid5", "kdd_iid10", "kdd_age3", "kdd_age5",
                        "acs_iid5", "acs_iid10", "acs_state3", "acs_state5",
                        "cac_iid5", "cac_iid10", "cac_state3", "cac_state5"
                    ],
                    help="Choose the dataset and split type")

args = parser.parse_args()

loader_map = {
   # Adult
    "adult_iid5": lambda: load_adult_random(num_clients=5),
    "adult_iid10": lambda: load_adult_random(num_clients=10),
    "adult_age3": load_adult_age3,
    "adult_age5": load_adult_age5,
    # Bank
    "bank_iid5": lambda: load_bank_random(num_clients=5),
    "bank_iid10": lambda: load_bank_random(num_clients=10),
    "bank_age3": load_bank_age3,
    "bank_age5": load_bank_age_5, 
    # KDD
    "kdd_iid5": lambda: load_kdd_random(num_clients=5),
    "kdd_iid10": lambda: load_kdd_random(num_clients=10),
    "kdd_age3": load_kdd_age3,
    "kdd_age5": load_kdd_age5,
    # ACS
    "acs_iid5": lambda: load_acs_random(num_clients=5),
    "acs_iid10": lambda: load_acs_random(num_clients=10),
    "acs_state3": load_acs_states_3,
    "acs_state5": load_acs_states_5,
    # CAC
    "cac_iid5": lambda: load_cac_random(num_clients=5),
    "cac_iid10": lambda: load_cac_random(num_clients=10),
    "cac_state3": load_cac_states_3,
    "cac_state5": load_cac_states_5,
}

print(f"DEBUG: Starting experiment with: {args.loader}")
raw_data = loader_map[args.loader]()

(client_data_dict, 
 X_test, y_test, s_test_list, col_names, y_test_pot, 
 X_val, y_val, s_val_list, y_val_pot) = raw_data

X_test = X_test.float().to(DEVICE)
y_test = y_test.float().to(DEVICE)
s_test = torch.tensor(s_test_list, dtype=torch.float32).to(DEVICE)

X_val = X_val.float().to(DEVICE)
y_val = y_val.float().to(DEVICE)
s_val = torch.tensor(s_val_list, dtype=torch.float32).to(DEVICE)

# Convert dictionary to list of tuples for the Server (X, y, A)
client_datasets = [
    (client["X"], client["y"], client["s"])
    for client in client_data_dict.values()
]

# Dynamically determine input size based on the loaded data
input_size = client_datasets[0][0].shape[1]
loss_function = BCELoss()

# -----------------------------
# 4. Initialize Server
# -----------------------------
server = Server(
    client_datasets=client_datasets,
    modelclass=build_model_class(input_size),
    lossf=loss_function,
    T=COMMUNICATION_ROUNDS,
    client_stepsize=CLIENT_STEPSIZE,
    client_batchsize=CLIENT_BATCHSIZE,
    client_epochs=CLIENT_EPOCHS,
    NY=NY,
    device=DEVICE,
    convergence=False,
    lambda_=LAMBDA 
)

# -----------------------------
# 5. Initialize Logger
# -----------------------------

log_config = {
    "CLIENT_EPOCHS": CLIENT_EPOCHS,
    "CLIENT_BATCHSIZE": CLIENT_BATCHSIZE,
    "CLIENT_STEPSIZE": CLIENT_STEPSIZE,
    "COMMUNICATION_ROUNDS": COMMUNICATION_ROUNDS,
    "LAMBDA": LAMBDA,
    "NY": NY
}

fl_logger = FLLogger(
    algorithm="Global_Group_Fairness_SP", 
    loader=args.loader,
    config=log_config
)

# -----------------------------
# 6. Define logging function for training progress
# -----------------------------

def print_log_progress():
    if not hasattr(print_log_progress, "current_round"):
        print_log_progress.current_round = 1

        print_log_progress.best_val_acc = -1.0
        print_log_progress.best_round_acc = -1
        print_log_progress.best_weights = None     
        
        print_log_progress.best_bal_acc = -1.0
        print_log_progress.best_round_bal_acc = -1
        
        print_log_progress.best_sp = float('inf')
        print_log_progress.best_round_sp = -1
        
        print_log_progress.best_eodd = float('inf')
        print_log_progress.best_round_eodd = -1

    server.model.eval() 
    with torch.no_grad():
        all_probs = server.model(X_val).flatten()
        
    all_labels = y_val.flatten()
    all_sensitive = s_val.flatten()

  
    all_logits = torch.logit(all_probs, eps=1e-6) 
    all_preds = (all_probs > 0.5).float()        
    
    # Accuracy
    acc = accuracy_score(all_labels.cpu(), all_preds.cpu())
    
    # Balanced Accuracy
    bal_acc = compute_balanced_accuracy(all_preds, all_labels)
    
    # Statistical Parity
    sp = compute_statistical_parity(all_preds, all_sensitive)
    
    # Equalized Odds
    eodd_val = compute_equalized_odds(all_preds, all_labels, all_sensitive)


    # Check high scores for everything ---
    if acc > print_log_progress.best_val_acc:
        print_log_progress.best_val_acc = acc
        print_log_progress.best_round_acc = print_log_progress.current_round
                # Deepcopy saves an exact, independent clone of the weights
        print_log_progress.best_weights = copy.deepcopy(server.model.state_dict())
        
    if bal_acc > print_log_progress.best_bal_acc:
        print_log_progress.best_bal_acc = bal_acc
        print_log_progress.best_round_bal_acc = print_log_progress.current_round
        
    if abs(sp) < print_log_progress.best_sp:
        print_log_progress.best_sp = abs(sp)
        print_log_progress.best_round_sp = print_log_progress.current_round
        
    if eodd_val < print_log_progress.best_eodd:
        print_log_progress.best_eodd = eodd_val
        print_log_progress.best_round_eodd = print_log_progress.current_round
    # ----------------------------------------------------------------------
    
    print(f"Round {print_log_progress.current_round}: Acc={acc:.4f}, BalAcc={bal_acc:.4f}, SP={sp:.4f}, Eodd={eodd_val:.4f}")

    # --- LOCAL CSV/JSON LOGGING ---
    fl_logger.log_round(
        round_idx=print_log_progress.current_round,
        metrics={
            "Accuracy": acc,
            "balanced_Accuracy": bal_acc,
            "Statistical_Parity": sp,
            "Equalized_Odds": eodd_val
        }
    )

    client_reports = []
    for i, client in enumerate(server.clients):
        with torch.no_grad():
            c_probs = server.model(client.X).flatten()
            c_labels = client.Y.flatten()
            c_loss = loss_function(c_probs, c_labels).item()
            
            client_reports.append({
                "client_name": f"Client_{i+1}",
                "loss": c_loss,
                "samples": len(c_labels)
            })
            
    fl_logger.log_clients(print_log_progress.current_round, client_reports)
    
    # Switch model back to train mode for the next round
    server.model.train()
    print_log_progress.current_round += 1

server.log_progress = print_log_progress


# -----------------------------
# 7. Train
# -----------------------------

# Initialize N and sensitive attribute stats
server.sync_N()

# Run the Federated Training Loop
server.train()

# -----------------------------
# 8. Evaluate Global Model
# -----------------------------
print("\n--- Final Global Evaluation ---")

if print_log_progress.best_weights is not None:
    server.model.load_state_dict(print_log_progress.best_weights)

server.model.eval()
with torch.no_grad():
    all_probs = server.model(X_test).flatten()

all_labels = y_test.flatten()
all_sensitive = s_test.flatten()
all_preds = (all_probs > 0.5).float() 

# Use your colleague's exact functions
final_acc = accuracy_score(all_labels.cpu(), all_preds.cpu())
final_bal = compute_balanced_accuracy(all_preds, all_labels)
final_sp = compute_statistical_parity(all_preds, all_sensitive)
final_eodd = compute_equalized_odds(all_preds, all_labels, all_sensitive)

print(f"Global Model Accuracy: {final_acc:.4f}")
print(f"Global Model Balanced Accuracy: {final_bal:.4f}")
print(f"Global Model Statistical Parity (SP): {final_sp:.4f}")
print(f"Global Model Equalized Odds (Eodd): {final_eodd:.4f}")

fl_logger.best_metrics = {
    "Accuracy": {"round": print_log_progress.best_round_acc, "value": print_log_progress.best_val_acc},
    "balanced_Accuracy": {"round": print_log_progress.best_round_bal_acc, "value": print_log_progress.best_bal_acc},
    "Statistical_Parity": {"round": print_log_progress.best_round_sp, "value": print_log_progress.best_sp},
    "Equalized_Odds": {"round": print_log_progress.best_round_eodd, "value": print_log_progress.best_eodd}
}

fl_logger.log_round(
    round_idx="FINAL_TEST",
    metrics={
        "Accuracy": final_acc,
        "balanced_Accuracy": final_bal,
        "Statistical_Parity": final_sp,
        "Equalized_Odds": final_eodd
    }
)

# Finaize the logger
fl_logger.finalize()

print("\nBest Rounds Summary:")
for metric, info in fl_logger.best_metrics.items():
    print(f"{metric}: Round {info['round']} | Value: {info['value']:.4f}")