import torch
import numpy as np
from sklearn.metrics import accuracy_score
from models import simpleModel #reuse the model definition
from fairnessMetrics import compute_equalized_odds, compute_statistical_parity, compute_balanced_accuracy

class FedAvgServer:
    def __init__(self, test_data, input_dim, device='cuda'):
        """
        Args:
            test_data (tuple): (X_test, y_test, s_test)
            input_dim (int): Feature size for the model
            device (str): 'cpu' or 'cuda'
        """
        self.device = device
        
        #Unpack and move test data to device
        self.X_test = test_data[0].to(self.device)
        self.y_test = test_data[1].to(self.device).view(-1, 1)
        #Convert sensitive attributes to tensor if they are a list
        if isinstance(test_data[2], list):
            self.s_test = torch.tensor(test_data[2], dtype=torch.float32).to(self.device).view(-1, 1)
        else:
            self.s_test = test_data[2].to(self.device).view(-1, 1)

        self.X_val = test_data[3].to(self.device)
        self.y_val = test_data[4].to(self.device).view(-1, 1)
        #Convert sensitive attributes to tensor if they are a list
        if isinstance(test_data[5], list):
            self.s_val = torch.tensor(test_data[5], dtype=torch.float32).to(self.device).view(-1, 1)
        else:
            self.s_val = test_data[5].to(self.device).view(-1, 1)

        #Initialize Global Model
        self.global_model = simpleModel(input_dim).to(self.device)
        
        self.trust_scores = {} 

        self.global_lambda = 0.0 

        self.device = device

    def loadBestModel(self, best_state_dict):
        "Loads the best model found during training (e.g., for final evaluation)."
        self.global_model.load_state_dict(best_state_dict)

    def broadcast_weights(self):
        """Returns the global model weights (on CPU) to be sent to clients."""
        return {k: v.cpu() for k, v in self.global_model.state_dict().items()}

    def aggregate(self, client_reports, agg_strategy):
        """
        Generic aggregation step.
        
        Args:
            client_reports (list): List of dicts returned by clients.
            agg_strategy (func): Function that computes new weights.
        """
        
        #Execute Strategy to get new weights
        new_weights = agg_strategy(client_reports, self.global_model, self.device)
        
        #Update Global Model
        self.global_model.load_state_dict(new_weights)




    def evaluate(self, final=False):
        """
        Runs inference on the global test set and computes metrics.
        Returns: Dict {Accuracy, SP, EO}
        """
        self.global_model.eval()
        if final:
            eval_X, eval_y, eval_s = self.X_test, self.y_test, self.s_test
        else:
            eval_X, eval_y, eval_s = self.X_val, self.y_val, self.s_val

        with torch.no_grad():
            logits = self.global_model(eval_X)
            preds = (logits > 0.5).float()
            
            # Accuracy
            acc = accuracy_score(eval_y.cpu(), preds.cpu())

            # Prepare tensors (Ensure they are flat 1D arrays)
            y_flat = eval_y.view(-1)
            s_flat = eval_s.view(-1)
            preds_flat = preds.view(-1)

            #Compute Fairness Metrics using new functions
            stat_parity = compute_statistical_parity(preds_flat, s_flat)
            eq_odds = compute_equalized_odds(preds_flat, y_flat, s_flat)
            balanced_acc = compute_balanced_accuracy(preds_flat, y_flat)

            return {
                "Accuracy": acc,
                "balanced_Accuracy": balanced_acc,
                "Statistical_Parity": stat_parity,
                "Equalized_Odds": eq_odds,
                "Lambda": self.global_lambda
            }