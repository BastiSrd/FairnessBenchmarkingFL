import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import simpleModel


class FedAvgClient:
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
        self.X = data_dict['X'].to(device)
        
        #Reshape y and s to (N, 1) to match model output shape
        self.y = data_dict['y'].to(device).view(-1, 1)
        self.s = data_dict['s'].to(device).view(-1, 1)
        
        #Create a DataLoader for batching (Standard SGD)
        dataset = TensorDataset(self.X, self.y, self.s)
        self.loader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        #Initialize Model
        self.model = simpleModel(input_dim).to(device)
        
        #Define Standard Criterion (Base Loss)
        self.criterion = nn.BCELoss()

    def set_parameters(self, global_weights):
        """
        Overwrites local model weights with the server's global weights.
        """
        self.model.load_state_dict(global_weights)

    def train(self, epochs, lr, loss_strategy, strategy_context=None):
        """
        Runs local training using the Strategy Pattern.
        
        Args:
            epochs (int): Number of local passes over data.
            lr (float): Learning rate.
            loss_strategy (func): Function that calculates loss (Standard or Fairness).
            strategy_context (dict): Extra data needed for the strategy (e.g., lambda).
        """
        if strategy_context is None:
            strategy_context = {}

        self.model.train()
        
        #Re-initialize optimizer every round to clear old momentum
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
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
            'samples': len(self.X)
        }