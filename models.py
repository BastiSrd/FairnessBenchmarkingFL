import torch
import torch.nn as nn


class modelFedMinMax(nn.Module):
    def __init__(self, input_dim):
        super(modelFedMinMax, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        
        self.layer2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
class simpleModel(nn.Module):
    def __init__(self, input_dim):
        super(simpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        
        self.layer2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x