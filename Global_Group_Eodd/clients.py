# +------------------------------------------------------------+
# |   clients.py                                                |
# |   Contains code executed by each client                    |
# |   Client abstraction is done via `Client` class            |
# +------------------------------------------------------------+

# imports
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def distance_kernel(a, b):
    return ((torch.abs(a - 1) + torch.abs(b - 1) - torch.abs(a - b)) + (torch.abs(a) + torch.abs(b) - torch.abs(a - b))) / 4


# Client Class
class Client:
    def __init__(self, dataset, model, lossf, stepsize=0.1, batchsize=100, epochs=10, lambda_= 1, device='cpu'):
        self.X = dataset[0].float().to(device)
        self.Y = dataset[1].float().to(device)
        self.A = dataset[2].float().to(device)
        self.stepsize = stepsize
        self.batchsize = batchsize
        self.epochs = epochs
        # --- MODIFICATION: Eodd needs alphas for 4 groups ---
        self.alpha_00 = None # A=0, Y=0
        self.alpha_10 = None # A=1, Y=0
        self.alpha_01 = None # A=0, Y=1
        self.alpha_11 = None # A=1, Y=1
        self.Y_test = None
        self.A_test = None
        self.Y0 = None
        self.Y1 = None
        self.model = model
        self.lossf = lossf
        self.lambda_ = lambda_
        self.K = lambda a, b: distance_kernel(a, b)
        self.current_C = lambda p, A, Y: 0
        self.device = device


    def client_step(self, current_theta):
        if self.batchsize is not None:
            generator = DataLoader(TensorDataset(self.X, self.Y, self.A), batch_size=self.batchsize, shuffle=True, num_workers=0)
        self.model.load_state_dict(current_theta)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.stepsize)
        for e in range(self.epochs):
            if self.batchsize is not None:
                for x, y, a in generator:
                    optimizer.zero_grad()
                    prediction = self.model(x)
                    accloss = self.lossf(prediction.flatten(), y)
                    # --- MODIFICATION: Eodd Loss (Conditional on Y) ---
                    # Gap for Y=0 (A=0 vs A=1)
                    fair_y0 = self.current_C(prediction[(a == 0) & (y == 0)], A=0, Y=0) - \
                              self.current_C(prediction[(a == 1) & (y == 0)], A=1, Y=0)
                    
                    # Gap for Y=1 (A=0 vs A=1)
                    fair_y1 = self.current_C(prediction[(a == 0) & (y == 1)], A=0, Y=1) - \
                              self.current_C(prediction[(a == 1) & (y == 1)], A=1, Y=1)
                    
                    loss = accloss + 2 * self.lambda_ * (fair_y0 + fair_y1)
                    
                    loss.backward()
                    optimizer.step()
            else:
                    optimizer.zero_grad()
                    prediction = self.model(self.X)
                    accloss = self.lossf(prediction.flatten(), self.Y)
                    
                    # --- MODIFICATION: Full Batch Eodd Logic ---
                    fair_y0 = self.current_C(prediction[(self.A == 0) & (self.Y == 0)], A=0, Y=0) - \
                              self.current_C(prediction[(self.A == 1) & (self.Y == 0)], A=1, Y=0)
                    
                    fair_y1 = self.current_C(prediction[(self.A == 0) & (self.Y == 1)], A=0, Y=1) - \
                              self.current_C(prediction[(self.A == 1) & (self.Y == 1)], A=1, Y=1)
                    
                    loss = accloss + 2 * self.lambda_ * (fair_y0 + fair_y1)
                    
                    loss.backward()
                    optimizer.step()
        # decrease learning rate
        self.stepsize = 0.99 * self.stepsize
        return self.model.state_dict()

    def sample_C_update(self, num_points, current_theta):
        # num_points is now [n00, n10, n01, n11]
        with torch.no_grad():
            self.model.load_state_dict(current_theta)
            # Helper to sample safely
            def get_preds(a_target, y_target, alpha, n_target):
                mask = (self.A == a_target) & (self.Y == y_target)
                count = mask.sum().item()
                if count == 0: 
                    return torch.tensor([], device=self.device)
                
                size = int(alpha * n_target)
                # Sample with replacement if we need more points than we have
                idxs = np.random.choice(count, size=max(1, size), replace=(size > count))
                return self.model(self.X[mask][idxs])

            p00 = get_preds(0, 0, self.alpha_00, num_points[0])
            p10 = get_preds(1, 0, self.alpha_10, num_points[1])
            p01 = get_preds(0, 1, self.alpha_01, num_points[2])
            p11 = get_preds(1, 1, self.alpha_11, num_points[3])
            
            return p00, p10, p01, p11

    def set_Y_sets(self, Y0, Y1):
        self.Y0 = Y0
        self.Y1 = Y1

    def set_N(self, N):
        self.N = N

    def get_weight(self):
        return len(self.Y)

    def test_client(self, theta):
        with torch.no_grad():
            self.model.load_state_dict(theta)
            return self.model(self.X_test), self.Y_test, self.A_test

    def split_train_test(self, **kwargs):
        self.X, self.X_test, self.Y, self.Y_test, self.A, self.A_test = train_test_split(self.X, self.Y, self.A, **kwargs)

        # --- MODIFICATION: Joint Probability P(A, Y) ---
    def get_Pkay(self, a, y):
        # Local probability P(A=a, Y=y)
        count = ((self.A == a) & (self.Y == y)).sum().float()
        return (count / len(self.Y)).cpu().item()

    def set_alpha_kay(self, Pa00, Pa10, Pa01, Pa11):
        def safe_div(n, d): return n/d if d > 0 else 0.0
        self.alpha_00 = safe_div(self.get_Pkay(0, 0), Pa00)
        self.alpha_10 = safe_div(self.get_Pkay(1, 0), Pa10)
        self.alpha_01 = safe_div(self.get_Pkay(0, 1), Pa01)
        self.alpha_11 = safe_div(self.get_Pkay(1, 1), Pa11)

    # --- MODIFICATION: Constraint Sets ---
    def set_C(self, Y_00, Y_10, Y_01, Y_11):
        def currentCfunction(p, A, Y):
            if len(p) == 0: return 0
            
            # Logic: Match A=0 to A=1 within same Y group
            if Y == 0:
                target_0 = Y_00.yset
                target_1 = Y_10.yset
                if A == 0:
                    return self.K(p, target_0).mean() * (self.N / (self.N - 1)) - self.K(p, target_1).mean()
                if A == 1:
                    return self.K(p, target_0).mean() - self.K(p, target_1).mean() * (self.N / (self.N - 1))
            
            if Y == 1:
                target_0 = Y_01.yset
                target_1 = Y_11.yset
                if A == 0:
                    return self.K(p, target_0).mean() * (self.N / (self.N - 1)) - self.K(p, target_1).mean()
                if A == 1:
                    return self.K(p, target_0).mean() - self.K(p, target_1).mean() * (self.N / (self.N - 1))
            return 0

        self.current_C = currentCfunction

