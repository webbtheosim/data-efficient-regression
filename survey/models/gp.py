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

        # Scale labels for NN training.
        self.label_scaler = StandardScaler()
        self.label_scaler.fit(y.reshape(-1,1))
        y_sc = self.label_scaler.transform(y.reshape(-1,1)).reshape(-1)

        # Convert to tensors.
        X = torch.tensor(X, dtype=torch.float64)
        y_sc = torch.tensor(y_sc, dtype=torch.float64)

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
            y_pred = self.label_scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(-1)

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
                y_std = self.label_scaler.inverse_transform(y_std.reshape(-1,1)).reshape(-1)

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
                    y_batch = self.label_scaler.inverse_transform(y_batch.reshape(-1,1)).reshape(-1)
                    y_std.append(y_batch)
            y_std = np.concatenate(y_std, axis=0)

        return y_std.reshape(-1)