import torch
import numpy as np
from sklearn.metrics import accuracy_score
from models import modelFedMinMax
from fairnessMetrics import compute_statistical_parity, compute_equalized_odds

class EOFedMinMaxServer:
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
        self.s_test_orig = torch.floor(self.s_test / 2).float()
        #Initialize Global Model
        self.global_model = modelFedMinMax(input_dim).to(self.device)
        self._avg_state = None   # dict[str, torch.Tensor]
        self._avg_t = 0          # how many rounds have been averaged


        self.trust_scores = {} 

        self.global_lambda = 0.0 

        self.device = device

        #FedMinMax
        self.lr_mu = 0.01 # Adversary learning rate


    def update_running_average(self, new_state_dict):
        """
        Update the running average:
        avg <- avg + (theta - avg)/t
        where t counts how many global models have been averaged so far.
        """
        # Move tensors to server device to keep everything consistent
        new_sd = {k: v.detach().to(self.device) for k, v in new_state_dict.items()}

        if self._avg_state is None:
            self._avg_state = {k: v.clone() for k, v in new_sd.items()}
            self._avg_t = 1
            return

        self._avg_t += 1
        t = self._avg_t
        for k, v in new_sd.items():
            self._avg_state[k].add_((v - self._avg_state[k]) / t)

    def get_averaged_state(self):
        """Return averaged params if available, else current model params."""
        if self._avg_state is None:
            return self.global_model.state_dict()
        return self._avg_state

    def load_averaged_model(self):
        """Load the averaged parameters into the global model (use only at the end)."""
        self.global_model.load_state_dict(self.get_averaged_state())

    #FedMinMax
    def set_global_stats(self, total_group_counts):
        """
        Call this from main.py after loading data to set accurate priors rho.
        total_group_counts: dict {gid: count}
        """
        self.real_group_counts = dict(total_group_counts)

        total = sum(total_group_counts.values())
        self.group_ids = sorted(total_group_counts.keys())  # true group ids, deterministic order
        self.gid_to_idx = {gid: i for i, gid in enumerate(self.group_ids)}
        rho_list = [total_group_counts[gid] / total for gid in self.group_ids]
        self.rho = torch.tensor(rho_list, device=self.device)
        self.mu = torch.tensor(rho_list, device=self.device) # Initialize mu = rho [cite: 236]

    def initializeWeights(self):
        """Returns the global model weights (on CPU) and group weights to be sent to clients."""

        # Avoid division by zero
        
        
        w_vector = self.mu / (self.rho + 1e-10) 
        # Convert to dict for easier client consumption
        w_dict = {gid: w_vector[self.gid_to_idx[gid]].item() for gid in self.group_ids}
        
        return {
            'model_weights': {k: v.cpu() for k, v in self.global_model.state_dict().items()},
            'group_weights': w_dict
        }

    def aggregate(self, client_reports, agg_strategy):
        """
        Generic aggregation step.
        
        Args:
            client_reports (list): List of dicts returned by clients.
            agg_strategy (func): Function that computes new weights.
        """
        '''old
        #Execute Strategy to get new weights
        new_weights = agg_strategy(client_reports, self.global_model, self.device)
        
        #Update Global Model
        self.global_model.load_state_dict(new_weights)
        '''
        #FedMinMax
        # Pack state needed for FedMinMax
        server_state = {
            'mu': self.mu,
            'rho': self.rho,
            'lr_mu': self.lr_mu,
            'group_counts': self.real_group_counts,   # exact n_a
            'group_ids': self.group_ids,              # exact order used for mu/rho
            'gid_to_idx': self.gid_to_idx
        }

        # Execute Strategy
        new_weights = agg_strategy(client_reports, self.global_model, self.device, server_state)

        # Store debug info if strategy set it
        if 'debug_risk_vector' in server_state:
            self._last_risk_vector = server_state['debug_risk_vector'].detach().clone()
        else:
            self._last_risk_vector = None
        
        # Update Global Model and average model for final output
        self.global_model.load_state_dict(new_weights)
        self.update_running_average(new_weights)

        # Update internal mu (Strategies might update state in place, but good to be explicit)
        self.mu = server_state['mu']

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
            s_orig_flat = self.s_test_orig.view(-1)
            preds_flat = preds.view(-1)

            # 2. Compute Fairness Metrics using new functions
            stat_parity = compute_statistical_parity(preds_flat, s_orig_flat)
            eq_odds = compute_equalized_odds(preds_flat, y_flat, s_orig_flat)

            return {
                "Accuracy": acc,
                "Statistical_Parity": stat_parity,
                "Equalized_Odds": eq_odds,
                "Lambda": self.global_lambda
            }
        

    def log_fedminmax_state(self, round_idx, risk_vector=None, top_k=5):
        """
        Logs mu, rho, w=mu/rho and optionally the per-group global risk_vector used to update mu.
        Assumes: self.group_ids, self.mu, self.rho, self.gid_to_idx exist.
        """
        with torch.no_grad():
            mu = self.mu.detach().cpu()
            rho = self.rho.detach().cpu()
            w = (mu / (rho + 1e-12)).cpu()

            gids = list(self.group_ids)

            # Build printable rows
            rows = []
            for i, gid in enumerate(gids):
                r = None
                if risk_vector is not None:
                    r = float(risk_vector[i].detach().cpu())
                rows.append((gid, float(rho[i]), float(mu[i]), float(w[i]), r))

            # Sort by risk (if available) else by w
            if risk_vector is not None:
                rows_sorted = sorted(rows, key=lambda x: (x[4] if x[4] is not None else -1e9), reverse=True)
                criterion = "risk"
            else:
                rows_sorted = sorted(rows, key=lambda x: x[3], reverse=True)
                criterion = "w"

            print(f"\n[Round {round_idx}] FedMinMax Server State")
            print(f"  Sum(mu)={mu.sum().item():.6f}  min(mu)={mu.min().item():.6f}  max(mu)={mu.max().item():.6f}")
            print(f"  Sum(rho)={rho.sum().item():.6f}  min(rho)={rho.min().item():.6f}  max(rho)={rho.max().item():.6f}")
            print(f"  w stats: min(w)={w.min().item():.6f}  max(w)={w.max().item():.6f}")

            header = "  gid |   rho    |    mu    |   w=mu/rho  "
            if risk_vector is not None:
                header += "|   risk"
            print(header)
            print("  " + "-" * (len(header)-2))

            for tup in rows_sorted[:top_k]:
                gid, rho_i, mu_i, w_i, r_i = tup
                if r_i is None:
                    print(f"  {gid:>3} | {rho_i:8.4f} | {mu_i:8.4f} | {w_i:10.4f}")
                else:
                    print(f"  {gid:>3} | {rho_i:8.4f} | {mu_i:8.4f} | {w_i:10.4f} | {r_i:7.4f}")
