import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import gpytorch
import torch
import pickle

import sys
sys.path.append('../survey')
from models import get_model

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RBFKernel
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

class ExactGPModel(ExactGP):
    '''
        Implementation of an exact GP model per GPyTorch tutorials.
    '''
    def __init__(self, train_x, train_y, likelihood, isotropic=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        if isotropic:
            self.covar_module = ScaleKernel(RBFKernel()) 
        else: 
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1])) 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class GP:
    '''
        GPyTorch implementation of a Gaussian process for guiding
        active learning campaigns. The GP kernel can be made isotropic
        or anisotropic upon specification.
    '''

    def __init__(self, isotropic=True):
        self.name = 'gp'
        self.isotropic = True

    def train(self, X, y, tune=False, train_iter=10000, print_progress=False):
        '''
            If tune is set to True, then kernel parameters are optimized according
            to the provided data. If tune is set to False, then the GP's reference
            data is updated without modifying kernel parameters.
        '''

        # Convert to tensors.
        X = torch.tensor(X, dtype=torch.float64)
        y_sc = torch.tensor(y, dtype=torch.float64)

        # Tune kernel parameters for improved fitting.
        if tune:

            # Prepare model for training.
            self.likelihood = GaussianLikelihood()
            self.model = ExactGPModel(X, y_sc, self.likelihood, self.isotropic)
            self.likelihood.train()
            self.model.train()

            # Optimize kernel parameters.
            losses = []
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
            mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
            for i in range(train_iter):

                optimizer.zero_grad()
                output = self.model(X)
                loss = -mll(output, y_sc.view(-1))
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                if i > 1000 and np.abs(loss.item() - losses[-100]) / np.abs(losses[-100]) < 1e-3:
                    print('Stopping early!')
                    break

                if print_progress and (i+1) % 100 == 0:
                    print('Iter %d/%d - Loss: %.3f | Lengthscale: %.3f | Noise: %.3f' % (
                        i + 1, train_iter, loss.item(),
                        self.model.covar_module.base_kernel.lengthscale.item(),
                        self.model.likelihood.noise.item()
                    ))

        # Adjust training data without modifying kernel parameters.
        else:
            self.model.set_train_data(X, y_sc, strict=False)

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float64)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X))
            y_pred = observed_pred.mean.detach().numpy()
        return y_pred

    def get_uncertainties(self, X):
        '''
            Breaks up uncertainty evaluations based on the size of X, since
            GPs tend to scale poorly past 10,000 evaluations at a time.
        '''
        self.model.eval()
        self.likelihood.eval()
        X = torch.tensor(X, dtype=torch.float64)
        if X.shape[0] < 10000:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(X))
                y_std = observed_pred.stddev.detach().numpy()
        else:
            batch_size = 1000
            if X.shape[0] % batch_size == 0:
                n_batches = int(X.shape[0] / batch_size)
            else:
                n_batches = int(X.shape[0] / batch_size) + 1
            y_std = []
            for batch_idx in range(n_batches):
                start = batch_size * batch_idx                
                end = min(X.shape[0], batch_size * (batch_idx + 1))
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = self.likelihood(self.model(X[start:end]))
                    y_batch = observed_pred.stddev.detach().numpy()
                    y_std.append(y_batch)
            y_std = np.concatenate(y_std, axis=0)
        return y_std.reshape(-1)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='esol')
    parser.add_argument('--model', default='gp_ard')
    args = parser.parse_args()

    for seed in range(1,11):

        # Input parameters.
        task_name = args.task
        path = f'{task_name}-sf-medoids-100-{seed}.npy'
        num = int(path.split('-')[3])

        # Load training data and task domain.
        task = pd.read_csv(f'../survey/tasks/{task_name}.csv', index_col=0)
        X = task.iloc[:,0:-1].to_numpy()
        y = task.iloc[:,-1].to_numpy()
        X = StandardScaler().fit_transform(X)
        chosen_idx = np.load(f'../survey/datasets/size_{num}/{path}')
        X_train = X[chosen_idx]
        y_train = y[chosen_idx]

        # Quantify support of the test data.
        gp_model = GP(isotropic=False)
        gp_model.train(X_train, y_train, tune=True)
        support = gp_model.get_uncertainties(X)

        # Train the model of interest and evaluate.
        if args.model != 'gp_ard':
            model = get_model(model_name=args.model)
            model.train(X_train, y_train, tune=True)
            y_pred = model.predict(X)
        else:
            y_pred = gp_model.predict(X)
        y_err = np.abs(y - y_pred)

        # # Quantify correlation of support and error.
        # plt.rcParams['font.size'] = 10
        # plt.rcParams['axes.linewidth'] = 1.1
        # fig, ax = plt.subplots(1,1,figsize=(3.5,3.5))
        # ax.hexbin(support, y_err, bins='log', cmap=plt.get_cmap('Blues'), gridsize=30)
        # ax.set_xlabel('GP Uncertainty')
        # ax.set_ylabel('Model Error')
        # ax.tick_params(width=1.1, length=3.3)
        # ax.spines[['top', 'right']].set_visible(False)
        # plt.tight_layout()
        # plt.savefig('./support_error.png', dpi=300)

        corr = spearmanr(support, y_err).statistic
        with open('./data/errors.csv', 'a') as handle:
            handle.write(f'{args.task},{args.model},{seed},{corr:.5f}\n')