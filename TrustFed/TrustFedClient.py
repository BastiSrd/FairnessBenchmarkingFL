import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models import simpleModel  # identisch wie FedAvg-Setup


class TrustFedClient:
    """
    Federated learning client for TrustFed.

    Extends the standard FedAvg client by allowing a loss strategy
    that includes fairness constraints (e.g., Demographic Parity or Equalized Odds).

    Expected keys in data_dict:
        - "X": Feature tensor, shape (N, D)
        - "y": Binary labels (0/1), shape (N,)
        - "s": Sensitive attribute values, shape (N,)
    """

    def __init__(self, client_name, data_dict, input_dim, device="cuda"):
        """
        Initialize a TrustFed client.

        Parameters:
            client_name (str): Unique identifier of the client.
            data_dict (dict): Must contain tensors "X", "y", and "s".
            input_dim (int): Input feature dimension for the model.
            device (str or torch.device): Device for computation ("cuda" or "cpu").
        """

        self.name = client_name
        self.device = device

        self.X = data_dict["X"].to(device)
        self.y = data_dict["y"].to(device).view(-1, 1)
        self.s = data_dict["s"].to(device).view(-1, 1)

        dataset = TensorDataset(self.X, self.y, self.s)
        self.loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model = simpleModel(input_dim).to(device)

        self.criterion = nn.BCELoss(reduction="none")

    def set_parameters(self, global_weights):
        self.model.load_state_dict(global_weights)

    def train(self, epochs, lr, loss_strategy, strategy_context=None):
        if strategy_context is None:
            strategy_context = {}

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        strategy_context["criterion"] = self.criterion
        strategy_context["device"] = self.device

        epoch_loss = 0.0
        epoch_constraint = 0.0

        for _ in range(epochs):
            batch_loss_sum = 0.0
            batch_constraint_sum = 0.0

            for batch_X, batch_y, batch_s in self.loader:
                optimizer.zero_grad()

                outputs = self.model(batch_X)

                loss_out = loss_strategy(outputs, batch_y, batch_s, {**strategy_context, "X_batch": batch_X})

                if isinstance(loss_out, tuple):
                    loss, constraint_loss = loss_out
                    batch_constraint_sum += float(constraint_loss.detach().cpu().item())
                else:
                    loss = loss_out

                loss.backward()

                # default values
                clip_norm = 1.0
                clip_norm_2 = 1.0
                epsilon = 3.0
                delta = 1.0

                noise_scale = (clip_norm_2 / epsilon) * ((2.0 * math.log(1.25 / delta)) ** 0.5)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    noise = torch.normal(mean=0.0, std=noise_scale, size=p.grad.shape, device=p.grad.device)
                    p.grad.add_(noise)

                optimizer.step()

                batch_loss_sum += float(loss.detach().cpu().item())

            epoch_loss += batch_loss_sum / max(1, len(self.loader))
            if len(self.loader) > 0:
                epoch_constraint += batch_constraint_sum / max(1, len(self.loader))

        avg_loss = epoch_loss / max(1, epochs)
        avg_constraint = epoch_constraint / max(1, epochs)

        print(f"{self.name} loss: {avg_loss}")

        return {
            "client_name": self.name,
            "weights": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "loss": avg_loss,
            "constraint_loss": avg_constraint,
            "samples": len(self.X),
        }
