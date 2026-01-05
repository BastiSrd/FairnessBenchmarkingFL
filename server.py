import torch
import numpy as np
from sklearn.metrics import accuracy_score
from client import simpleModel #reuse the model definition

class FLServer:
    def __init__(self, test_data, input_dim, device='cpu'):
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

        #Initialize Global Model
        self.global_model = simpleModel(input_dim).to(self.device)
        
        self.trust_scores = {} 

        self.global_lambda = 0.0 

        self.device = device

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


    def get_rates(self, preds, labels, sensitive, s_value):
        """
        Helper to calculate TPR, FPR, and PPR.
        Args:
            preds (torch.Tensor): Binary predictions (0 or 1) of shape (N,).
            labels (torch.Tensor): Ground truth labels (0 or 1) of shape (N,). 
                                   Can be None if only PPR is needed.
            sensitive (torch.Tensor): Sensitive attribute values of shape (N,).
            s_value (int or float): The specific value in 'sensitive' that defines 
                                    the group to calculate metrics for (e.g., 0).

        Returns:
            tuple: A tuple containing (TPR, FPR, PPR) as floats. 
                   Returns (0.0, 0.0, 0.0) if the group is empty.
        """
        mask = (sensitive == s_value)
        
        if mask.sum() == 0:
            return 0.0, 0.0, 0.0

        # Filter predictions
        g_preds = preds[mask]

        # --- Positive Prediction Rate (PPR) ---
        ppr = g_preds.float().mean().item()

        if labels is None:
            return 0.0, 0.0, ppr

        # Filter labels
        g_labels = labels[mask]

        # --- True Positive Rate (TPR) ---
        mask_pos = (g_labels == 1)
        if mask_pos.sum() > 0:
            tpr = g_preds[mask_pos].float().mean().item()
        else:
            tpr = 0.0

        # --- False Positive Rate (FPR) ---
        mask_neg = (g_labels == 0)
        if mask_neg.sum() > 0:
            fpr = g_preds[mask_neg].float().mean().item()
        else:
            fpr = 0.0

        return tpr, fpr, ppr

    def compute_statistical_parity(self, preds, sensitive):
        """
        Calculates Statistical Parity: P(Pred=1 | Non-Prot) - P(Pred=1 | Prot)

        Args:
            preds (torch.Tensor): Binary predictions (0 or 1) of shape (N,).
            sensitive (torch.Tensor): Sensitive attribute values of shape (N,). 
                                      Must contain exactly two unique values 
                                      (e.g., 0 and 1) to calculate a difference.

        Returns:
            float: The signed difference in acceptance rates. Positive values 
                   indicate Group B is accepted more often than Group A.
                   Returns 0.0 if fewer than 2 groups are found.
        """
        groups = torch.unique(sensitive).sort()[0]
        if len(groups) < 2:
            return 0.0
        
        group_a = groups[0].item()
        group_b = groups[1].item()
        
        _, _, ppr_a = self.get_rates(preds, None, sensitive, group_a)
        _, _, ppr_b = self.get_rates(preds, None, sensitive, group_b)
        
        return ppr_b - ppr_a

    def compute_equalized_odds(self, preds, labels, sensitive):
        """
        Calculates the Equalized Odds score, defined as the average absolute 
        difference in True Positive Rates (TPR) and False Positive Rates (FPR) 
        between two groups.

        Formula: 0.5 * (|TPR_A - TPR_B| + |FPR_A - FPR_B|)

        Args:
            preds (torch.Tensor): Binary predictions (0 or 1) of shape (N,).
            labels (torch.Tensor): Ground truth labels (0 or 1) of shape (N,).
            sensitive (torch.Tensor): Sensitive attribute values of shape (N,). 
                                      Must contain exactly two unique values.

        Returns:
            float: The Equalized Odds gap.
                   Returns 0.0 if fewer than 2 groups are found.
        """
        groups = torch.unique(sensitive).sort()[0]
        
        if len(groups) < 2:
            return 0.0
            
        group_a = groups[0].item()
        group_b = groups[1].item()
        
        # Get rates for both groups
        tpr_a, fpr_a, _ = self.get_rates(preds, labels, sensitive, group_a)
        tpr_b, fpr_b, _ = self.get_rates(preds, labels, sensitive, group_b)

        # Calculate EO
        tpr_diff = abs(tpr_b - tpr_a)
        fpr_diff = abs(fpr_b - fpr_a)

        return 0.5 * (tpr_diff + fpr_diff)

    def evaluate(self):
        """
        Runs inference on the global test set and computes metrics.
        Returns: Dict {Accuracy, SP, EO}
        """
        self.global_model.eval()
        with torch.no_grad():
            logits = self.global_model(self.X_test)
            preds = (logits > 0.5).float()
            
            # Accuracy
            acc = accuracy_score(self.y_test.cpu(), preds.cpu())

            # Prepare tensors (Ensure they are flat 1D arrays)
            y_flat = self.y_test.view(-1)
            s_flat = self.s_test.view(-1)
            preds_flat = preds.view(-1)

            # 2. Compute Fairness Metrics using new functions
            stat_parity = self.compute_statistical_parity(preds_flat, s_flat)
            eq_odds = self.compute_equalized_odds(preds_flat, y_flat, s_flat)

            return {
                "Accuracy": acc,
                "Statistical_Parity": stat_parity,
                "Equalized_Odds": eq_odds,
                "Lambda": self.global_lambda
            }