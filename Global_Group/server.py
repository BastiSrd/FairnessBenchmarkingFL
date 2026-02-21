# +------------------------------------------------------------+
# |   server.py                                                |
# |   Contains code executed on the server                     |
# |   Server abstraction is done via `Server` class            |
# +------------------------------------------------------------+

# imports
import numpy as np
import torch
from tqdm import tqdm
import wandb

from . import clients
from . import yset
from . import metrics


def copy_statedict(statedict, device='cpu'):
    statedict_copy = {}
    for key in statedict.keys():
        statedict_copy[key] = torch.zeros_like(statedict[key], device=device)
        statedict_copy[key] += statedict[key]
    return statedict_copy


# Server Class
class Server:
    def __init__(self, client_datasets, modelclass, lossf, m=None, T=50, client_stepsize=5e-2, client_batchsize=100, client_epochs=10, mu=1.0, NY=100, lambda_=1, datasetname='None', runname='', device='cpu', convergence=False, additional_config={}):
        '''
        Initializes the Server object for federated learning experiment.

        Args:
            client_datasets (list): A list of (X,Y,A) tuples, one for each client, where X,Y,A are torch tensors.
            modelclass (function handle): Function which returns a model
            lossf (function handle): loss function to use
            m (int or None, optional): The number of clients to use in each communication round. If set to None, all clients will participate in each communication round. Default is None.
            T (int, optional): The total number of communication rounds. Must be a positive integer. Default is 50.
            client_stepsize (float, optional): The stepsize used by each client for local updates. Default is 0.1.
            client_batchsize (int, optional): The batchsize used for client updates. Default is 100.
            client_epochs (int, optional): The number of epochs each client does per communication round. Default is 10.
        '''
        self.m = len(client_datasets) if m is None else m
        self.T = T
        self.clients = [clients.Client(dataset, modelclass().to(device), lossf, stepsize=client_stepsize, batchsize=client_batchsize, epochs=client_epochs, lambda_=lambda_, device=device) for dataset in client_datasets]
        self.client_weights = [client.get_weight() for client in self.clients]
        self.client_weights = np.array(self.client_weights) / sum(self.client_weights)
        self.model = modelclass().to(device)
        self.mu = mu
        self.Y_0, self.Y_1 = None, None
        self.device = device
        self.convergence = convergence
        config={
            "m": m,
            "T": T,
            "client_epochs": client_epochs,
            "client_stepsize": client_stepsize,
            "client_batchsize": client_batchsize,
            "mu": mu,
            "NY": NY,
            "lambda_": lambda_,
            "dataset": datasetname,
            "runname": runname
        }
        config.update(additional_config)

        self.NY = NY
        if self.convergence:
            wandb.init(
                # set the wandb project where this run will be logged
                project="fairFLConvergence",
    
                # track hyperparameters and run metadata
                config={
                    "lambda_": lambda_,
                    "dataset": datasetname,
                }
            )
        else:
            wandb.init(
                # set the wandb project where this run will be logged
                project="fairFL",
    
                # track hyperparameters and run metadata
                config = config
            )

    def aggregate_theta(self, thetas, weights):
        global_state_dict = {}
        for key in self.model.state_dict().keys():
            global_state_dict[key] = torch.zeros_like(self.model.state_dict()[key], device=self.device)

        # Compute the weighted average of local models' state dictionaries
        for i, local_model in enumerate(thetas):
            for key in local_model.keys():
                global_state_dict[key] += local_model[key] * weights[i]

        # Update the global model's state dictionary
        self.model.load_state_dict(global_state_dict)

    def client_step(self):
        '''
        Performs the client steps for the participating clients in a single communication round.

        This method selects a random subset of clients to participate in the communication round based on the value of `self.m`. Then, for each participating client, it performs local steps
        on the client's dataset using the specified hyperparameters. Finally, the method returns a list of models, where each model is the result of the local update step performed by a participating client.

        Returns:
            A list of models, where each model is the result of the local update step performed by a participating client.

        '''
        participating_client_ids = self.sample_clients()
        return [self.clients[id].client_step(copy_statedict(self.model.state_dict(), device=self.device)) for id in participating_client_ids], [self.client_weights[id] for id in participating_client_ids]

    def sample_clients(self):
        '''
        Selects a random subset of clients to participate in the current communication round.
        This method selects `m` clients at random from the list of `client_datasets`. The value of `m` is determined by the `m` attribute of the `Server` object. If `m` is `None`, all clients are selected.

        Returns:
            A list of participating clients.

        '''
        round_client_ids = np.random.choice(len(self.client_weights), self.m, p=self.client_weights, replace=False)
        return round_client_ids

    def train(self, callback=None):
        '''
        Trains the federated learning model.

        This method trains the federated learning model using the specified hyperparameters and datasets. The training is performed over a fixed number of communication rounds.
        '''
        # sync the Pa
        self.sync_Pa()
        # init the sets for C
        self.Y_0, self.Y_1 = yset.YSet(0, self.NY, device=self.device), yset.YSet(1, self.NY, device=self.device)
        # perform the communication rounds
        for t in tqdm(range(self.T)):
            # update the C function (algorithm 2)
            self.update_C()
            for client in self.clients:
                client.set_C(self.Y_0, self.Y_1)
            # perform client updates (algorithm 3)
            model_updates, update_weights = self.client_step()
            self.aggregate_theta(model_updates, update_weights)
            if self.convergence:
                self.log_all()
                #self.log_losses()
            else:
                self.log_progress()
        wandb.finish()

    def update_C(self):
        # drop samples
        self.Y_0.drop(self.mu)
        self.Y_1.drop(self.mu)

        # sample new points
        updates0 = []
        updates1 = []
        for client, weight in zip(self.clients, self.client_weights):
            p0, p1 = client.sample_C_update([int(weight * self.mu * self.NY)] * 2, copy_statedict(self.model.state_dict(), device=self.device))
            updates0.append(p0)
            updates1.append(p1)
        self.Y_0.update(updates0)
        self.Y_1.update(updates1)

    def test_current_model(self):
        '''
        PLACEHOLDER
        '''
        client_predictions = [client.test_client(copy_statedict(self.model.state_dict(), device=self.device)) for client in self.clients]
        return client_predictions, self.client_weights

    def train_test_split(self, fraction=0.25):
        '''
        Performs a train-test split on each client's dataset.

        This method instructs each client to perform a train-test split on their dataset using the specified fraction for the test set. The train-test split is performed randomly, and the same split is used for each communication round.

        Args:
            fraction (fload, optional): The fraction of the samples to use for the test set. This should be a value between 0 and 1. Default is 0.25.
        '''
        for client in self.clients:
            client.split_train_test(test_size=fraction)
        self.client_weights = [client.get_weight() for client in self.clients]
        self.client_weights = np.array(self.client_weights) / sum(self.client_weights)

    def sync_Pa(self):
        Pk0s = [client.get_Pka(a=0) for client in self.clients]
        Pa0 = (np.array(Pk0s) * self.client_weights).sum()
        for client in self.clients:
            client.set_alphaka(Pa0)

    def log_progress(self):
        res, weights = self.test_current_model()
        acc = metrics.accuracy(
            torch.cat([r[0] for r in res]).flatten(),
            torch.cat([r[1] for r in res]).flatten()
        )
        fairness = metrics.P1(
            torch.cat([r[0] for r in res]).flatten(),
            torch.cat([r[2] for r in res]).flatten()
        )
        wandb.log({"acc": acc.cpu(), "fairness": fairness.cpu()})

    def sync_N(self):
        N = sum((client.get_weight() for client in self.clients))
        for client in self.clients:
            client.set_N(N)

    def log_losses(self):
        def distance_kernel(a, b):
            return ((torch.abs(a - 1) + torch.abs(b - 1) - torch.abs(a - b)) + (torch.abs(a) + torch.abs(b) - torch.abs(a - b))) / 4

        def MMD(a, b):
            return (
                distance_kernel(a, a.T).mean()
                + distance_kernel(b, b.T).mean()
                - 2 * distance_kernel(a, b.T).mean()
            )

        with torch.no_grad():
            X = torch.cat([client.X for client in self.clients])
            Y = torch.cat([client.Y for client in self.clients])
            A = torch.cat([client.A for client in self.clients])
            Y_hat = self.model(X)
            train_accloss = torch.nn.BCEWithLogitsLoss()(Y_hat.flatten(), Y).cpu()
            train_fairloss = MMD(Y_hat[A == 0], Y_hat[A == 1])

            X = torch.cat([client.X_test for client in self.clients])
            Y = torch.cat([client.Y_test for client in self.clients])
            A = torch.cat([client.A_test for client in self.clients])
            Y_hat = self.model(X)
            test_accloss = torch.nn.BCEWithLogitsLoss()(Y_hat.flatten(), Y).cpu()
            test_fairloss = MMD(Y_hat[A == 0], Y_hat[A == 1]).cpu()

        wandb.log({"train_acc": train_accloss, 
                   "train_fair": train_fairloss,
                   "test_acc": test_accloss, 
                   "test_fair": test_fairloss,
                  })
    
    def log_all(self):
        def distance_kernel(a, b):
            return ((torch.abs(a - 1) + torch.abs(b - 1) - torch.abs(a - b)) + (torch.abs(a) + torch.abs(b) - torch.abs(a - b))) / 4

        def MMD(a, b):
            return (
                distance_kernel(a, a.T).mean()
                + distance_kernel(b, b.T).mean()
                - 2 * distance_kernel(a, b.T).mean()
            )

        with torch.no_grad():
            X = torch.cat([client.X for client in self.clients])
            Y = torch.cat([client.Y for client in self.clients])
            A = torch.cat([client.A for client in self.clients])
            Y_hat = self.model(X)
            train_accloss = torch.nn.BCEWithLogitsLoss()(Y_hat.flatten(), Y).cpu()
            train_fairloss = MMD(Y_hat[A == 0], Y_hat[A == 1])

            X = torch.cat([client.X_test for client in self.clients])
            Y = torch.cat([client.Y_test for client in self.clients])
            A = torch.cat([client.A_test for client in self.clients])
            Y_hat = self.model(X)
            test_accloss = torch.nn.BCEWithLogitsLoss()(Y_hat.flatten(), Y).cpu()
            test_fairloss = MMD(Y_hat[A == 0], Y_hat[A == 1]).cpu()
        
        res, weights = self.test_current_model()
        acc = metrics.accuracy(
            torch.cat([r[0] for r in res]).flatten(),
            torch.cat([r[1] for r in res]).flatten()
        )
        fairness = metrics.P1(
            torch.cat([r[0] for r in res]).flatten(),
            torch.cat([r[2] for r in res]).flatten()
        )

        wandb.log({"train_acc": train_accloss, 
                   "train_fair": train_fairloss,
                   "test_acc": test_accloss, 
                   "test_fair": test_fairloss,
                   "acc": acc.cpu(),
                   "fairness": fairness.cpu()
                  })
