import math
import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

import functools
print = functools.partial(print, flush=True)

def load_dataset(task_name, n_max):
    task = pd.read_csv(f'../tasks/{task_name}.csv', index_col=0)
    X = task.iloc[:,0:-1].to_numpy()
    y = task.iloc[:,-1].to_numpy()
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1,1)).reshape(-1)
    if X.shape[0] > n_max:
        idx = [i for i in range(X.shape[0])]
        np.random.shuffle(idx)
        train_idx = idx[0:1000]
        X = X[train_idx]
        y = y[train_idx]
    return X, y

class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 100))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(100, 2))

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
            num_dims=2, grid_size=100
        )
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DeepGP:
    '''
        Wrapper for a Gaussian process with a neural network feature extractor,
        all trained end-to-end.
    '''
    def __init__(self):
        super().__init__()
        self.feature_extractor = None
        self.model = None
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    def fit(self, X, y, max_epochs=1000):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.feature_extractor = FeatureExtractor(data_dim=X.shape[1])
        self.model = GPRegressionModel(X, y, self.likelihood, self.feature_extractor)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': self.model.feature_extractor.parameters()},
            {'params': self.model.covar_module.parameters()},
            {'params': self.model.mean_module.parameters()},
            {'params': self.model.likelihood.parameters()},
        ], lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        patience = 30
        min_improvement = 0.01
        best_loss = float("inf")
        epochs_since_improvement = 0
        for i in range(max_epochs):
            print(f'Training on epoch: {i+1}/{max_epochs}', end=' | ')
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.backward()
            train_loss = loss.item()
            print(f'Loss = {train_loss:.5f}')
            relative_improvement = (best_loss - train_loss) / best_loss
            if relative_improvement > min_improvement:
                best_loss = train_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(
                    f"Stopping early: loss has not improved by more than "
                    f"{100*min_improvement:.1f}% for {patience} epochs."
                )
                break
            optimizer.step()
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X))
            y_pred = observed_pred.mean.detach().numpy()
        return y_pred

class GP:
    '''Wrapper for simple Exact GP'''
    def __init__(self):
        super().__init__()
        self.model = None
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    def fit(self, X, y, max_epochs=1000):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.model = ExactGP(X, y, self.likelihood)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        patience = 30
        min_improvement = 0.01
        best_loss = float("inf")
        epochs_since_improvement = 0
        for i in range(max_epochs):
            print(f'Training on epoch: {i+1}/{max_epochs}', end=' | ')
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.backward()
            train_loss = loss.item()
            print(f'Loss = {train_loss:.5f}')
            relative_improvement = (best_loss - train_loss) / best_loss
            if relative_improvement > min_improvement:
                best_loss = train_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(
                    f"Stopping early: loss has not improved by more than "
                    f"{100*min_improvement:.1f}% for {patience} epochs."
                )
                break
            optimizer.step()
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X))
            y_pred = observed_pred.mean.detach().numpy()
        return y_pred

def evaluate_model(X, y, model):
    maes = []
    kf = KFold(n_splits=5)
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f'Training for split {i+1}/5')
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        model.fit(X_train, y_train, max_epochs=100)
        y_pred = model.predict(X_test)
        mae = np.mean(np.abs(y_test - y_pred))
        maes.append(mae)
    return np.mean(maes), np.std(maes)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='esol')
    parser.add_argument('--n_max', type=int, default=1000)
    args = parser.parse_args()

    X, y = load_dataset(args.task, args.n_max)
    model = DeepGP()
    mae1, err1 = evaluate_model(X, y, model)
    print(f'{mae1:.3f} +/- {err1:.3f}')
    model = GP()
    mae2, err2 = evaluate_model(X, y, model)
    print(f'{mae2:.3f} +/- {err2:.3f}')
    err_ratio = mae1 / mae2
    print(f'Error ratio: {err_ratio}')
    with open('./data/stationary.csv', 'a') as handle:
        handle.write(f'{args.task},{mae1},{err1},{mae2},{err2},{err_ratio}\n')