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
    def __init__(self, dataset, model, lossf, stepsize=0.1, batchsize=100, epochs=10, lambda_=1, device='cpu'):
        self.X = dataset[0].float().to(device)
        self.Y = dataset[1].float().to(device)
        self.A = dataset[2].float().to(device)
        self.stepsize = stepsize
        self.batchsize = batchsize
        self.epochs = epochs
        self.alphak0 = None
        self.alphak1 = None
        self.Y_test = None
        self.A_test = None
        self.Y0 = None
        self.Y1 = None
        self.model = model
        self.lossf = lossf
        self.lambda_ = lambda_
        self.K = lambda a, b: distance_kernel(a, b)
        self.current_C = lambda p: 0
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
                    fairloss = self.current_C(prediction[a == 0], A=0) - self.current_C(prediction[a == 1], A=1)
                    loss = accloss + 2 * self.lambda_ * fairloss
                    
                    loss.backward()
                    optimizer.step()
            else:
                    optimizer.zero_grad()
                    prediction = self.model(self.X)
                    accloss = self.lossf(prediction.flatten(), self.Y)
                    fairloss = self.current_C(prediction[self.A == 0], A=0) - self.current_C(prediction[self.A == 1], A=1)
                    loss = accloss + 2 * self.lambda_ * fairloss
                    loss.backward()
                    optimizer.step()
        # decrease learning rate
        self.stepsize = 0.99 * self.stepsize
        return self.model.state_dict()

    def sample_C_update(self, num_points, current_theta):
        with torch.no_grad():
            self.model.load_state_dict(current_theta)
            point_idxs_A1 = np.random.choice(int(self.A.sum()), size=int(self.alphak1 * num_points[1]), replace=False)
            point_idxs_A0 = np.random.choice(int((1 - self.A).sum()), size=int(self.alphak0 * num_points[0]), replace=False)
            p1 = self.model(self.X[self.A == 1][point_idxs_A1])
            p0 = self.model(self.X[self.A == 0][point_idxs_A0])
            return p0, p1

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

    def get_Pka(self, a=0):
        return (self.A == a).to(float).mean().cpu()

    def set_alphaka(self, Pa0):
        self.alphak0 = self.get_Pka(a=0) / Pa0
        self.alphak1 = self.get_Pka(a=1) / (1 - Pa0)

    def set_C(self, Y_0, Y_1):
        def currentCfunction(p, A=None):
            if len(p) == 0:
                return 0
            else:
                if A is None:
                    return self.K(p, Y_0.yset).mean() - self.K(p, Y_1.yset).mean()
                if A == 1:
                    return self.K(p, Y_0.yset).mean() - self.K(p, Y_1.yset).mean() * (self.N / (self.N - 1))
                if A == 0:
                    return self.K(p, Y_0.yset).mean() * (self.N / (self.N - 1)) - self.K(p, Y_1.yset).mean()
        self.current_C = currentCfunction

