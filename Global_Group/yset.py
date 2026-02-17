# +------------------------------------------------------------+
# |   yset.py                                                  |
# |   Contains code for syncing the Y sets                     |
# |   Y_0 and Y_1 abstraction is done via `YSet` class         |
# +------------------------------------------------------------+

# imports
import torch
import numpy as np


class YSet:
    def __init__(self, A, NY, device='cpu'):
        self.A = A
        self.yset = torch.zeros(NY, device=device)

    def drop(self, mu):
        keep_idxs = np.random.choice(len(self.yset), int((1 - mu) * len(self.yset)), replace=False)
        self.yset = self.yset[keep_idxs]

    def update(self, yset):
        updates = torch.cat([update.flatten() for update in yset], 0)
        self.yset = torch.cat((self.yset, updates), 0)
