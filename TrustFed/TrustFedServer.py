import torch
from sklearn.metrics import accuracy_score

from fairnessMetrics import compute_equalized_odds, compute_statistical_parity, compute_balanced_accuracy
from models import simpleModel


class TrustFedServer:
    """
    Central server for the TrustFed framework.

    Maintains the global model, aggregates client updates,
    and evaluates performance and fairness on the test set.
    """

    def __init__(self, test_data, input_dim, device="cpu", fairness_notion="SP", alpha=1.0):
        self.device = device

        self.X_test = test_data[0].to(self.device)
        self.y_test = test_data[1].to(self.device).view(-1, 1)

        if isinstance(test_data[2], list):
            self.s_test = torch.tensor(test_data[2], dtype=torch.float32).to(self.device).view(-1, 1)
        else:
            self.s_test = test_data[2].to(self.device).view(-1, 1)

        self.global_model = simpleModel(input_dim).to(self.device)

        self.fairness_notion = fairness_notion  # "SP" or "EO"
        self.alpha = float(alpha)

    def broadcast_weights(self):
        return {k: v.cpu() for k, v in self.global_model.state_dict().items()}

    def aggregate(self, client_reports, agg_strategy):
        new_weights = agg_strategy(client_reports, self.global_model, self.device)
        self.global_model.load_state_dict(new_weights)

    def evaluate(self):
        self.global_model.eval()
        with torch.no_grad():
            logits = self.global_model(self.X_test)
            preds = (logits > 0.5).float()

            y_flat = self.y_test.view(-1)
            s_flat = self.s_test.view(-1)
            preds_flat = preds.view(-1)

            acc = accuracy_score(self.y_test.cpu(), preds.cpu())
            stat_parity = compute_statistical_parity(preds_flat, s_flat)
            eq_odds = compute_equalized_odds(preds_flat, y_flat, s_flat)
            balanced_accuracy = compute_balanced_accuracy(preds_flat, y_flat)

            return {
                "Accuracy": acc,
                "Statistical_Parity": stat_parity,
                "Equalized_Odds": eq_odds,
                "Balanced_Accuracy": balanced_accuracy,
                "Alpha": self.alpha,
                "FairnessNotion": self.fairness_notion,
            }
