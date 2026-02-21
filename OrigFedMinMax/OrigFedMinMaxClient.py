import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import modelFedMinMax

class OrigFedMinMaxClient:
    def __init__(self, client_name, data_dict, input_dim, device='cuda'):
        """
        Args:
            client_name (str): Identifier (e.g., "client_1").
            data_dict (dict): From loader {'X': tensor, 'y': tensor, 's': tensor, ...}
            input_dim (int): Number of features in X.
            device (str): 'cpu' or 'cuda'.
        """
        self.name = client_name
        self.device = device
        
        #move data to the device immediately to avoid transfer overhead later
        self.X = data_dict['X'].to(device).float()
        
        #Reshape y and s to (N, 1) to match model output shape
        self.y = data_dict['y'].to(device).float().view(-1, 1)
        self.s_original = data_dict['s'].to(device).view(-1, 1)
        self.s = (self.s_original * 2 + self.y).long() # This is now our "group"
        
        #Create a DataLoader for batching (Standard SGD)
        dataset = TensorDataset(self.X, self.y, self.s)
        self.loader = DataLoader(dataset, batch_size=len(self.X), shuffle=True)
        
        #Initialize Model
        self.model = modelFedMinMax(input_dim).to(device)
        
        #Define Standard Criterion (Base Loss)
        self.criterion = nn.BCEWithLogitsLoss()

    def set_parameters(self, global_weights):
        """
        Overwrites local model weights with the server's global weights.
        """
        self.model.load_state_dict(global_weights)

    def evaluate_group_risks(self):
        """
        Calculates risk (loss) per sensitive group on the CURRENT model weights.
        Required for FedMinMax step 9.
        """
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss(reduction='sum') # Sum to aggregate easier later
        
        group_risks = {}
        group_counts = {}
        
        # Identify unique groups in local data
        unique_groups = torch.unique(self.s)
        
        with torch.no_grad():
            # Entire dataset forward pass (okay for small/medium data, batch if memory issues)
            outputs = self.model(self.X)
            
            for gid in unique_groups:
                gid = gid.item()
                mask = (self.s == gid).squeeze()
                if mask.sum() == 0: continue
                
                loss = criterion(outputs[mask], self.y[mask])
                count = mask.sum().item()
                
                group_risks[gid] = (loss / count).item() # Average risk
                group_counts[gid] = count
                
        return group_risks, group_counts

    def train(self, epochs, lr, loss_strategy, strategy_context):
        """
        Runs local training using the Strategy Pattern.
        
        Args:
            epochs (int): Number of local passes over data.
            lr (float): Learning rate.
            loss_strategy (func): Function that calculates loss (Standard or Fairness).
            strategy_context (dict): Extra data needed for the strategy (e.g., lambda).
        """


        initial_group_risks, group_counts = self.evaluate_group_risks()
        
        self.model.train()
        
        #Re-initialize optimizer every round to clear old momentum
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        
        #Inject dependencies into context for the strategy function
        strategy_context['criterion'] = self.criterion
        strategy_context['device'] = self.device
        
        epoch_loss = 0.0
        
        for epoch in range(epochs):
            batch_loss_sum = 0
            for batch_X, batch_y, batch_s in self.loader:
                
                optimizer.zero_grad()
                
                # Forward Pass
                outputs = self.model(batch_X)
                
                loss = loss_strategy(outputs, batch_y, batch_s, strategy_context)
                
                loss.backward()
                optimizer.step()
                
                batch_loss_sum += loss.item()
            
            #Average loss for this epoch
            epoch_loss += batch_loss_sum / len(self.loader)
        print(f"{self.name} loss: {epoch_loss / epochs}")

        #Return a report dictionary to the Server
        return {
            'client_name': self.name,
            'weights': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'loss': epoch_loss / epochs,
            'samples': len(self.X),
            # New fields for FedMinMax
            'group_risks': initial_group_risks,
            'group_counts': group_counts
        }