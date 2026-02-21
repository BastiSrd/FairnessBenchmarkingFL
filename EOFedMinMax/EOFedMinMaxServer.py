import torch
import numpy as np
from sklearn.metrics import accuracy_score
from models import modelFedMinMax
from fairnessMetrics import compute_statistical_parity, compute_equalized_odds, compute_balanced_accuracy


class EOFedMinMaxServer:
    def __init__(self, test_data, input_dim, device='cpu'):
        self.device = device

        self.X_test = test_data[0].to(self.device).float()
        self.y_test = test_data[1].to(self.device).float().view(-1, 1)

        if isinstance(test_data[2], list):
            self.s_test = torch.tensor(test_data[2], dtype=torch.long).to(self.device).view(-1, 1)
        else:
            self.s_test = test_data[2].to(self.device).long().view(-1, 1)

        self.x_val = test_data[3].to(self.device).float()
        self.y_val = test_data[4].to(self.device).float().view(-1, 1)

        if isinstance(test_data[5], list):
            self.s_val = torch.tensor(test_data[5], dtype=torch.long).to(self.device).view(-1, 1)
        else:
            self.s_val = test_data[5].to(self.device).long().view(-1, 1)

        self.global_model = modelFedMinMax(input_dim).to(self.device)

        # running average of iterates
        self._avg_state = None
        self._avg_t = 0

        # adaptive lambda
        self.global_lambda = 3.0
        self.lambda_lr = 1.5      # how fast lambda adapts
        self.lambda_max = 5.0     # cap for stability
        self.eo_target = 0.00001     # target EO (you can tune)

        # adversary LR
        self.lr_mu = 0.05

        # will be set by set_global_stats
        self.group_ids = None
        self.gid_to_idx = None
        self.rho_y0 = None
        self.rho_y1 = None
        self.mu_y0 = None
        self.mu_y1 = None
        self.real_count_y0 = None
        self.real_count_y1 = None

    def update_running_average(self, new_state_dict):
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
        if self._avg_state is None:
            return self.global_model.state_dict()
        return self._avg_state

    def load_averaged_model(self):
        self.global_model.load_state_dict(self.get_averaged_state())

    def set_global_stats(self, total_counts_by_group):
        """
        Expect:
          total_counts_by_group = {
              gid: {"y0": count_neg, "y1": count_pos},
              ...
          }
        """
        # normalize keys
        real_y0 = {}
        real_y1 = {}
        for gid, v in total_counts_by_group.items():
            gid_int = int(gid)
            real_y0[gid_int] = int(v.get("y0", 0))
            real_y1[gid_int] = int(v.get("y1", 0))

        self.real_count_y0 = real_y0
        self.real_count_y1 = real_y1

        self.group_ids = sorted(set(real_y0.keys()) | set(real_y1.keys()))
        self.gid_to_idx = {gid: i for i, gid in enumerate(self.group_ids)}

        total0 = sum(real_y0.values())
        total1 = sum(real_y1.values())

        # Priors per label-slice
        rho0 = [(real_y0.get(gid, 0) / (total0 + 1e-12)) for gid in self.group_ids]
        rho1 = [(real_y1.get(gid, 0) / (total1 + 1e-12)) for gid in self.group_ids]

        self.rho_y0 = torch.tensor(rho0, device=self.device, dtype=torch.float32)
        self.rho_y1 = torch.tensor(rho1, device=self.device, dtype=torch.float32)

        # init mu = rho
        self.mu_y0 = self.rho_y0.clone()
        self.mu_y1 = self.rho_y1.clone()

    def _update_lambda_from_val(self):
        """
        Adaptive lambda update based on validation EO (hard preds).
        Increase lambda if EO is above target, decrease otherwise.
        """
        self.global_model.eval()
        with torch.no_grad():
            logits = self.global_model(self.x_val)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

        y_flat = self.y_val.view(-1)
        s_flat = self.s_val.view(-1)
        p_flat = preds.view(-1)

        eo = float(compute_equalized_odds(p_flat, y_flat, s_flat))

        # proportional control
        err = eo - self.eo_target
        
        self.global_lambda = float(np.clip(self.global_lambda + self.lambda_lr * err, 0.001, self.lambda_max))
      

    def initializeWeights(self):
        """
        Returns:
          - model_weights
          - group_weights_y0 (for y==0 slice)
          - group_weights_y1 (for y==1 slice)
          - lambda_eo (adaptive)
        """
        # avoid division by zero
        w0 = self.mu_y0 / (self.rho_y0 + 1e-10)
        w1 = self.mu_y1 / (self.rho_y1 + 1e-10)

        w0_dict = {gid: float(w0[self.gid_to_idx[gid]].item()) for gid in self.group_ids}
        w1_dict = {gid: float(w1[self.gid_to_idx[gid]].item()) for gid in self.group_ids}

        return {
            'model_weights': {k: v.cpu() for k, v in self.global_model.state_dict().items()},
            'group_weights_y0': w0_dict,
            'group_weights_y1': w1_dict,
            'lambda_eo': float(self.global_lambda),
        }

    def aggregate(self, client_reports, agg_strategy):
        server_state = {
            'mu_y0': self.mu_y0,
            'mu_y1': self.mu_y1,
            'rho_y0': self.rho_y0,
            'rho_y1': self.rho_y1,
            'lr_mu': self.lr_mu,
            'group_counts_y0': self.real_count_y0,
            'group_counts_y1': self.real_count_y1,
            'group_ids': self.group_ids,
            'gid_to_idx': self.gid_to_idx,
        }

        new_weights = agg_strategy(client_reports, self.global_model, self.device, server_state)

        self.global_model.load_state_dict(new_weights)
        self.update_running_average(new_weights)

        # pull back updated mus
        self.mu_y0 = server_state['mu_y0']
        self.mu_y1 = server_state['mu_y1']

        # update lambda from current validation EO
        self._update_lambda_from_val()

    def evaluate(self, final=False):
        self.global_model.eval()

        if final:
            eval_X, eval_y, eval_s = self.X_test, self.y_test, self.s_test
        else:
            eval_X, eval_y, eval_s = self.x_val, self.y_val, self.s_val

        with torch.no_grad():
            logits = self.global_model(eval_X)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            acc = accuracy_score(eval_y.cpu(), preds.cpu())

            y_flat = eval_y.view(-1)
            s_flat = eval_s.view(-1)
            preds_flat = preds.view(-1)

            stat_parity = compute_statistical_parity(preds_flat, s_flat)
            eq_odds = compute_equalized_odds(preds_flat, y_flat, s_flat)
            balAcc = compute_balanced_accuracy(preds_flat, y_flat)

            return {
                "Accuracy": acc,
                "balanced_Accuracy": balAcc,
                "Statistical_Parity": stat_parity,
                "Equalized_Odds": eq_odds,
                "Lambda": self.global_lambda
            }