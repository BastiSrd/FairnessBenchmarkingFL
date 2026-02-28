import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import modelFedMinMax


class EOFedMinMaxClient:
    def __init__(self, client_name, data_dict, input_dim, device='cuda'):
        self.name = client_name
        self.device = device

        # Ensure consistent dtypes/shapes
        self.X = data_dict['X'].to(device).float()
        self.y = data_dict['y'].to(device).float().view(-1, 1)
        self.s = data_dict['s'].to(device).long().view(-1, 1)

        dataset = TensorDataset(self.X, self.y, self.s)
        
        self.loader = DataLoader(dataset, batch_size=len(self.X), shuffle=True)

        self.model = modelFedMinMax(input_dim).to(device)

        self.criterion = nn.BCEWithLogitsLoss()

    def set_parameters(self, global_weights):
        self.model.load_state_dict(global_weights)

    def evaluate_group_eo_stats(self):
        """
        Compute group-wise EO stats on CURRENT model:
        - soft FPR proxy: mean(p) over y==0
        - soft FNR proxy: mean(1-p) over y==1
        Returns dicts + label-split counts.
        """
        self.model.eval()

        group_fpr = {}
        group_fnr = {}
        count_y0 = {}
        count_y1 = {}

        s = self.s.view(-1).long()
        y = self.y.view(-1).float()

        unique_groups = torch.unique(s)

        with torch.no_grad():
            logits = self.model(self.X).view(-1).float()
            p = torch.sigmoid(logits)

            for gid in unique_groups:
                gid_int = int(gid.item())
                mask_g = (s == gid_int)
                if not mask_g.any():
                    continue

                m0 = mask_g & (y < 0.5)
                n0 = int(m0.sum().item())
                if n0 > 0:
                    group_fpr[gid_int] = float(p[m0].mean().item())
                    count_y0[gid_int] = n0
                else:
                    count_y0[gid_int] = 0

                m1 = mask_g & (y >= 0.5)
                n1 = int(m1.sum().item())
                if n1 > 0:
                    group_fnr[gid_int] = float((1.0 - p[m1].mean()).item())
                    count_y1[gid_int] = n1
                else:
                    count_y1[gid_int] = 0

        return group_fpr, group_fnr, count_y0, count_y1

    def train(self, epochs, lr, loss_strategy, strategy_context):
        
        group_fpr, group_fnr, count_y0, count_y1 = self.evaluate_group_eo_stats()

        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # Inject dependencies into context
        strategy_context['criterion'] = self.criterion
        strategy_context['device'] = self.device

        epoch_loss = 0.0
        for _ in range(epochs):
            batch_loss_sum = 0.0
            for batch_X, batch_y, batch_s in self.loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = loss_strategy(outputs, batch_y, batch_s, strategy_context, device=self.device)
                loss.backward()
                optimizer.step()
                batch_loss_sum += loss.item()
            epoch_loss += batch_loss_sum / len(self.loader)

        print(f"{self.name} loss: {epoch_loss / epochs}")

        return {
            'client_name': self.name,
            'weights': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'loss': epoch_loss / epochs,
            'samples': len(self.X),

            # EO stats for server adversary update
            'group_fpr': group_fpr,
            'group_fnr': group_fnr,
            'count_y0': count_y0,
            'count_y1': count_y1
        }